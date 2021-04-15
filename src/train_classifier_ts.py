import argparse
import datetime
import models
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
from config import cfg
from data import fetch_dataset, make_data_loader, separate_dataset_ts, make_batchnorm_dataset_ts, make_stats_batchnorm
from metrics import Metric
from modules import Teacher, Student
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
from logger import make_logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
for k in cfg:
    cfg[k] = args[k]
if args['control_name']:
    cfg['control'] = {k: v for k, v in zip(cfg['control'].keys(), args['control_name'].split('_'))} \
        if args['control_name'] != 'None' else {}
cfg['control_name'] = '_'.join(
    [cfg['control'][k] for k in cfg['control'] if cfg['control'][k]]) if 'control' in cfg else ''


def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['model_name'], cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    teacher_dataset = fetch_dataset(cfg['data_name'])
    student_dataset = fetch_dataset(cfg['student_data_name'])
    process_dataset(teacher_dataset)
    teacher_dataset['train'], student_dataset['train'], data_separate = separate_dataset_ts(teacher_dataset['train'],
                                                                                            student_dataset['train'])
    batchnorm_dataset = make_batchnorm_dataset_ts(teacher_dataset['train'], student_dataset['train'])
    teacher_data_loader = make_data_loader(teacher_dataset, cfg['model_name'])
    student_data_loader = make_data_loader(student_dataset, cfg['model_name'])
    teacher_model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    student_model = eval('models.{}().to(cfg["device"])'.format(cfg['student_model_name']))
    teacher_optimizer = make_optimizer(teacher_model, cfg['model_name'])
    teacher_scheduler = make_scheduler(teacher_optimizer, cfg['model_name'])
    student_optimizer = make_optimizer(student_model, cfg['student_model_name'])
    student_scheduler = make_scheduler(student_optimizer, cfg['student_model_name'])
    if cfg['resume_mode'] == 1:
        result = resume(cfg['model_tag'])
        last_epoch = result['epoch']
        if last_epoch > 1:
            teacher_model.load_state_dict(result['teacher_model_state_dict'])
            teacher_optimizer.load_state_dict(result['optimizer_state_dict'])
            teacher_scheduler.load_state_dict(result['scheduler_state_dict'])
            student_model.load_state_dict(result['student_model_state_dict'])
            student_optimizer.load_state_dict(result['student_optimizer_state_dict'])
            student_scheduler.load_state_dict(result['student_scheduler_state_dict'])
            teacher_logger = result['teacher_logger']
            student_logger = result['student_logger']
        else:
            teacher_logger = make_logger('output/runs/teacher_train_{}'.format(cfg['model_tag']))
            student_logger = make_logger('output/runs/student_train_{}'.format(cfg['model_tag']))
    else:
        last_epoch = 1
        teacher_logger = make_logger('output/runs/teacher_train_{}'.format(cfg['model_tag']))
        student_logger = make_logger('output/runs/student_train_{}'.format(cfg['model_tag']))
    teacher_metric = Metric({'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']})
    student_metric = Metric({'train': ['Loss'], 'test': ['Loss', 'Accuracy']})
    if cfg['world_size'] > 1:
        teacher_model = torch.nn.DataParallel(teacher_model, device_ids=list(range(cfg['world_size'])))
        student_model = torch.nn.DataParallel(student_model, device_ids=list(range(cfg['world_size'])))
    teacher = Teacher(teacher_model, student_model, cfg['student_threshold'])
    student = Student(teacher_model, student_model, cfg['student_threshold'])
    for epoch in range(last_epoch, cfg[cfg['model_name']]['num_epochs'] + 1):
        teacher_logger.safe(True)
        student_logger.safe(True)
        train(teacher_data_loader['train'], teacher, teacher_optimizer, teacher_metric, teacher_logger, epoch, 'T')
        test_teacher_model = make_stats_batchnorm(batchnorm_dataset, teacher_model, cfg['model_name'])
        test(teacher_data_loader['test'], test_teacher_model, teacher_metric, teacher_logger, epoch, 'T')
        student.sync()
        train(student_data_loader['train'], student, student_optimizer, student_metric, student_logger, epoch, 'S')
        test_student_model = make_stats_batchnorm(batchnorm_dataset, student_model, cfg['model_name'])
        test(teacher_data_loader['test'], test_student_model, student_metric, student_logger, epoch, 'S')
        teacher_scheduler.step()
        student_scheduler.step()
        teacher_logger.safe(False)
        student_logger.safe(False)
        teacher_model_state_dict = test_teacher_model.module.state_dict() if cfg['world_size'] > 1 else \
            test_teacher_model.state_dict()
        student_model_state_dict = test_student_model.module.state_dict() if cfg['world_size'] > 1 else \
            test_student_model.state_dict()
        result = {'cfg': cfg, 'epoch': epoch + 1, 'data_separate': data_separate,
                  'teacher_model_state_dict': teacher_model_state_dict,
                  'student_model_state_dict': student_model_state_dict,
                  'teacher_optimizer_state_dict': teacher_optimizer.state_dict(),
                  'student_optimizer_state_dict': student_optimizer.state_dict(),
                  'teacher_scheduler_state_dict': teacher_scheduler.state_dict(),
                  'student_scheduler_state_dict': student_scheduler.state_dict(),
                  'teacher_logger': teacher_logger, 'student_logger': student_logger}
        save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if teacher_metric.compare(teacher_logger.mean['test/{}'.format(teacher_metric.pivot_name)]):
            teacher_metric.update(teacher_logger.mean['test/{}'.format(teacher_metric.pivot_name)])
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        teacher_logger.reset()
        student_logger.reset()
    teacher_logger.safe(False)
    student_logger.safe(False)
    return


def train(data_loader, model, optimizer, metric, logger, epoch, tag):
    model.train_(True)
    start_time = time.time()
    for i, input in enumerate(data_loader):
        input = collate(input)
        input_size = input['data'].size(0)
        input = to_device(input, cfg['device'])
        optimizer.zero_grad()
        output = model(input)
        if output['loss'] is None:
            continue
        else:
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            output['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            evaluation = metric.evaluate(metric.metric_name['train'], input, output)
            logger.append(evaluation, 'train', n=input_size)
            if i % int((len(data_loader) * cfg['log_interval']) + 1) == 0:
                batch_time = (time.time() - start_time) / (i + 1)
                lr = optimizer.param_groups[0]['lr']
                epoch_finished_time = datetime.timedelta(seconds=round(batch_time * (len(data_loader) - i - 1)))
                exp_finished_time = epoch_finished_time + datetime.timedelta(
                    seconds=round((cfg[cfg['model_name']]['num_epochs'] - epoch) * batch_time * len(data_loader)))
                info = {'info': ['Model({}): {}'.format(tag, cfg['model_tag']),
                                 'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * i / len(data_loader)),
                                 'Learning rate: {:.6f}'.format(lr),
                                 'Epoch Finished Time: {}'.format(epoch_finished_time),
                                 'Experiment Finished Time: {}'.format(exp_finished_time)]}
                logger.append(info, 'train', mean=False)
                print(logger.write('train', metric.metric_name['train']))
    return


def test(data_loader, model, metric, logger, epoch, tag):
    with torch.no_grad():
        model.train(False)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            output = model(input)
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            evaluation = metric.evaluate(metric.metric_name['test'], input, output)
            logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model({}): {}'.format(tag, cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
    return


if __name__ == "__main__":
    main()
