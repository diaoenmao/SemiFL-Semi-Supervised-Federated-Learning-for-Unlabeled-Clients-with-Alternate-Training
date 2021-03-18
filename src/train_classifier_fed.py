import argparse
import copy
import datetime
import models
import numpy as np
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset
from config import cfg
from data import fetch_dataset, make_data_loader, split_dataset, separate_dataset, make_stats_batchnorm_fed
from fed import Federation
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
        teacher_control_name = '1_1_none_{}'.format(cfg['augment'])
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
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    local_optimizer = make_optimizer(model, 'local')
    global_optimizer = make_optimizer(model, 'global')
    local_scheduler = make_scheduler(local_optimizer, 'local')
    global_scheduler = make_scheduler(global_optimizer, 'global')
    metric = Metric({'train': {'Local': ['Local-Loss', 'Local-Accuracy']},
                     'test': {'Local': ['Local-Loss', 'Local-Accuracy'], 'Global': ['Global-Loss', 'Global-Accuracy']}})
    teacher_dataset['train'], student_dataset['train'] = separate_dataset(teacher_dataset['train'],
                                                                          student_dataset['train'],
                                                                          cfg['supervise_rate'])
    data_split, target_split = split_dataset(student_dataset, cfg['num_users'], cfg['data_split_mode'])
    if cfg['resume_mode'] == 1:
        result = resume(cfg['model_tag'])
        last_epoch = result['epoch']
        logger = result['logger']
        if last_epoch > 1:
            model.load_state_dict(result['model_state_dict'])
            local_optimizer.load_state_dict(result['local_optimizer_state_dict'])
            local_scheduler.load_state_dict(result['local_scheduler_state_dict'])
            global_optimizer.load_state_dict(result['global_optimizer_state_dict'])
            global_scheduler.load_state_dict(result['global_scheduler_state_dict'])
    else:
        last_epoch = 1
        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        logger_path = 'output/runs/train_{}_{}'.format(cfg['model_tag'], current_time)
        logger = Logger(logger_path)
    federation = Federation(model.state_dict(), global_optimizer.state_dict(), data_split, target_split)
    for epoch in range(last_epoch, cfg['global']['num_epochs'] + 1):
        logger.safe(True)
        train_teacher(teacher_dataset['train'], model, local_optimizer, metric, logger, epoch)
        test_model = make_stats_batchnorm_fed(teacher_dataset['train'], student_dataset['train'], model, 'a', 'global')
        student_dataset['train'] = make_student_dataset(student_dataset['train'], test_model)
        train_student(student_dataset['train'], federation, model, global_optimizer, local_optimizer, metric, logger,
                      epoch)
        test_model = make_stats_batchnorm_fed(teacher_dataset['train'], student_dataset['train'], model, 'b', 'global')
        test(teacher_dataset['test'], test_model, metric, logger, epoch)
        if cfg['local']['scheduler_name'] == 'ReduceLROnPlateau':
            local_scheduler.step(metrics=logger.mean['train/{}'.format(cfg['pivot_metric'])])
        else:
            local_scheduler.step()
        if cfg['global']['scheduler_name'] == 'ReduceLROnPlateau':
            global_scheduler.step(metrics=logger.mean['train/{}'.format(cfg['pivot_metric'])])
        else:
            global_scheduler.step()
        logger.safe(False)
        model_state_dict = test_model.state_dict()
        result = {'cfg': cfg, 'epoch': epoch + 1, 'data_split': data_split, 'target_split': target_split,
                  'model_state_dict': model_state_dict, 'local_optimizer_state_dict': local_optimizer.state_dict(),
                  'local_scheduler_state_dict': local_scheduler.state_dict(),
                  'global_optimizer_state_dict': global_optimizer.state_dict(),
                  'global_scheduler_state_dict': global_scheduler.state_dict(), 'logger': logger}
        save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    logger.safe(False)
    return


def train_teacher(dataset, model, optimizer, metric, logger, epoch):
    data_loader = make_data_loader({'train': dataset['train']}, 'local')['train']
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
            print(logger.write('train', metric.metric_name['train']['Local']))
    return


def train_student(dataset, federation, model, global_optimizer, local_optimizer, metric, logger, epoch):
    local, local_parameters, user_idx = make_local(dataset, federation)
    num_active_users = len(local)
    cfg['local']['lr'] = local_optimizer.param_groups[0]['lr']
    start_time = time.time()
    for m in range(num_active_users):
        local_parameters[m] = copy.deepcopy(local[m].train(local_parameters[m], metric, logger))
        if m % int((num_active_users * cfg['log_interval']) + 1) == 0:
            local_time = (time.time() - start_time) / (m + 1)
            epoch_finished_time = datetime.timedelta(seconds=local_time * (num_active_users - m - 1))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['global']['num_epochs'] - epoch) * local_time * num_active_users))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * m / num_active_users),
                             'ID: {}({}/{})'.format(user_idx[m], m + 1, num_active_users),
                             'Learning rate: {}'.format(cfg['local']['lr']),
                             'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']['Local']))
    federation.update(model, global_optimizer, local_parameters)
    return


def make_student_dataset(student_dataset, teacher_model):
    with torch.no_grad():
        student_data_loader = make_data_loader(student_dataset, 'global', shuffle={'train': False, 'test': False})
        teacher_model.train(False)
        train_target = []
        for i, input in enumerate(student_data_loader['train']):
            input = collate(input)
            input['target'] = None
            input = to_device(input, cfg['device'])
            output = teacher_model(input)
            train_target.append(output['target'])
        train_target = torch.cat(train_target, dim=0).cpu().numpy()
        test_target = []
        for i, input in enumerate(student_data_loader['test']):
            input = collate(input)
            input['target'] = None
            input = to_device(input, cfg['device'])
            output = teacher_model(input)
            test_target.append(output['target'])
        test_target = torch.cat(test_target, dim=0).cpu().numpy()
        student_dataset['train'].target = train_target
        student_dataset['test'].target = test_target
    return student_dataset


def test(dataset, model, metric, logger, epoch):
    with torch.no_grad():
        model.train(False)
        data_loader = make_data_loader({'test': dataset}, 'global')['test']
        for i, input in enumerate(data_loader):
            input = collate(input)
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            output = model(input)
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            evaluation = metric.evaluate(metric.metric_name['test']['Global'], input, output)
            logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']['Global']))
    return


def make_local(dataset, federation):
    num_active_users = int(np.ceil(cfg['active_rate'] * cfg['num_users']))
    user_idx = torch.arange(cfg['num_users'])[torch.randperm(cfg['num_users'])[:num_active_users]].tolist()
    local_parameters = federation.distribute(user_idx)
    local = [None for _ in range(num_active_users)]
    for m in range(num_active_users):
        data_loader_m = make_data_loader({
            'train': Subset(dataset, federation.data_split['train'][user_idx[m]])}, 'local')['train']
        local[m] = Local(data_loader_m, federation.target_split[user_idx[m]])
    return local, local_parameters, user_idx


class Local:
    def __init__(self, data_loader, target_split):
        self.data_loader = data_loader
        self.target_split = target_split

    def train(self, local_parameters, metric, logger):
        model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        model.load_state_dict(local_parameters)
        model.train(True)
        optimizer = make_optimizer(model, 'local')
        for local_epoch in range(1, cfg['local']['num_epochs'] + 1):
            for i, input in enumerate(self.data_loader):
                input = collate(input)
                input_size = input['data'].size(0)
                input['target_split'] = torch.tensor(self.target_split)
                input = to_device(input, cfg['device'])
                optimizer.zero_grad()
                output = model(input)
                output['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                evaluation = metric.evaluate(metric.metric_name['train']['Local'], input, output)
                logger.append(evaluation, 'train', n=input_size)
        local_parameters = model.state_dict()
        return local_parameters


if __name__ == "__main__":
    main()
