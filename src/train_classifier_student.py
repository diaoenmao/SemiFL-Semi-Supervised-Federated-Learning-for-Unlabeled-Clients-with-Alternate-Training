import argparse
import copy
import datetime
import models
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset
from config import cfg
from data import fetch_dataset, make_data_loader, separate_dataset_ts, make_stats_batchnorm, make_stats_batchnorm_ts, \
    make_student_dataset, make_teacher_dataset
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
from logger import Logger

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
        teacher_control_name = '1_1_none_{}_{}_none_none'.format(cfg['augment'], cfg['supervise_rate'])
        teacher_model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['model_name'], teacher_control_name]
        cfg['teacher_model_tag'] = '_'.join([x for x in teacher_model_tag_list if x])
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
    teacher_model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    teacher_result = resume(cfg['teacher_model_tag'], load_tag='best')
    teacher_model.load_state_dict(teacher_result['model_state_dict'])
    init_model_state_dict = teacher_result['init_model_state_dict']
    data_separate = teacher_result['data_separate']
    teacher_dataset['train'], student_dataset['train'] = separate_dataset_ts(teacher_dataset['train'],
                                                                             student_dataset['train'], data_separate)
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    model.load_state_dict(teacher_result['init_model_state_dict'])
    optimizer = make_optimizer(model, cfg['model_name'])
    scheduler = make_scheduler(optimizer, cfg['model_name'])
    metric = Metric({'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']})
    if cfg['resume_mode'] == 1:
        result = resume(cfg['model_tag'])
        last_epoch = result['epoch']
        logger = result['logger']
        if last_epoch > 1:
            model.load_state_dict(result['model_state_dict'])
            optimizer.load_state_dict(result['optimizer_state_dict'])
            scheduler.load_state_dict(result['scheduler_state_dict'])
    else:
        last_epoch = 1
        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        logger_path = 'output/runs/train_{}_{}'.format(cfg['model_tag'], current_time)
        logger = Logger(logger_path)
    if cfg['world_size'] > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(cfg['world_size'])))
    test_model = make_stats_batchnorm_ts(teacher_dataset['train'], student_dataset['train'], teacher_model,
                                         cfg['model_name'])
    test(teacher_dataset['test'], test_model, metric, logger, 0)
    student_dataset['train'] = make_student_dataset(student_dataset['train'], test_model)
    teacher_dataset['train'] = make_teacher_dataset(teacher_dataset['train'])
    for epoch in range(last_epoch, cfg[cfg['model_name']]['num_epochs'] + 1):
        logger.safe(True)
        train(teacher_dataset['train'], student_dataset['train'], model, optimizer, metric, logger, epoch)
        test_model = make_stats_batchnorm_ts(teacher_dataset['train'], student_dataset['train'], model,
                                             cfg['model_name'])
        test(teacher_dataset['test'], test_model, metric, logger, epoch)
        scheduler.step()
        logger.safe(False)
        model_state_dict = test_model.module.state_dict() if cfg['world_size'] > 1 else test_model.state_dict()
        result = {'cfg': cfg, 'epoch': epoch + 1, 'init_model_state_dict': init_model_state_dict,
                  'model_state_dict': model_state_dict, 'optimizer_state_dict': optimizer.state_dict(),
                  'scheduler_state_dict': scheduler.state_dict(), 'logger': logger}
        save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    logger.safe(False)
    return


def train(teacher_dataset, student_dataset, model, optimizer, metric, logger, epoch):
    dataset = ConcatDataset([teacher_dataset, student_dataset])
    data_loader = make_data_loader({'train': dataset}, cfg['model_name'])['train']
    model.train(True)
    start_time = time.time()
    for i, input in enumerate(data_loader):
        input = collate(input)
        input_size = input['data'].size(0)
        input = to_device(input, cfg['device'])
        optimizer.zero_grad()
        output = model(input)
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
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * i / len(data_loader)),
                             'Learning rate: {:.6f}'.format(lr), 'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']))
    return


def test(teacher_dataset, model, metric, logger, epoch):
    data_loader = make_data_loader({'test': teacher_dataset}, cfg['model_name'])['test']
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
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
    return


if __name__ == "__main__":
    main()
