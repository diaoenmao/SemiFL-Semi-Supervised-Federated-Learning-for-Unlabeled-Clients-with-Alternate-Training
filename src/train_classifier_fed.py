import argparse
import datetime
import models
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from config import cfg
from data import fetch_dataset, split_dataset, make_data_loader, separate_dataset, separate_dataset_cu, \
    make_batchnorm_dataset_cu, make_batchnorm_stats
from metrics import Metric
from modules import Center, User
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
        teacher_model_name = '1_1_none_{}_none_none'.format(cfg['num_supervised'])
        teacher_model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['model_name'], teacher_model_name]
        cfg['teacher_model_tag'] = '_'.join([x for x in teacher_model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    center_dataset = fetch_dataset(cfg['data_name'])
    user_dataset = fetch_dataset(cfg['user_data_name'])
    process_dataset(center_dataset)
    result = resume(cfg['teacher_model_tag'], load_tag='checkpoint')
    data_separate = result['data_separate']
    center_dataset['train'], user_dataset['train'], data_separate = separate_dataset_cu(center_dataset['train'],
                                                                                        user_dataset['train'],
                                                                                        data_separate)
    data_loader = make_data_loader(center_dataset, 'center')
    batchnorm_dataset = make_batchnorm_dataset_cu(center_dataset['train'], user_dataset['train'])
    data_split, target_split = split_dataset(user_dataset, cfg['num_users'], cfg['data_split_mode'])
    last_epoch = 1
    logger = make_logger('output/runs/train_{}'.format(cfg['model_tag']))
    metric = Metric({'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']})
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    center = make_center(center_dataset, model)
    user = make_user(user_dataset, data_split, model, cfg['threshold'])
    for epoch in range(last_epoch, cfg['global']['num_epochs'] + 1):
        center.distribute(user)
        train_center(center, metric, logger, epoch)
        logger.reset()
        train_user(user, metric, logger, epoch)
        center.update(user)
        model.load_state_dict(center.model_state_dict)
        test_model = make_batchnorm_stats(batchnorm_dataset, model, 'center')
        test(data_loader['test'], test_model, metric, logger, epoch)
        model_state_dict = model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict()
        result = {'cfg': cfg, 'epoch': epoch + 1, 'model_state_dict': model_state_dict,
                  'data_separate': data_separate, 'data_split': data_split,
                  'target_split': target_split, 'logger': logger}
        save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    return


def make_center(center_dataset, model):
    center = Center(center_dataset, model)
    return center


def make_user(user_dataset, data_split, model, threshold):
    user_id = torch.arange(cfg['num_users'])
    user = [None for _ in range(cfg['num_users'])]
    for m in range(len(user)):
        user_dataset_m = {'train': separate_dataset(user_dataset['train'], data_split['train'][m])[0],
                          'test': separate_dataset(user_dataset['test'], data_split['test'][m])[0]}
        user[m] = User(user_id[m], user_dataset_m, model, threshold)
    return user


def train_center(center, metric, logger, epoch):
    logger.safe(True)
    start_time = time.time()
    center.train(metric, logger)
    _time = (time.time() - start_time)
    lr = center.optimizer_state_dict['param_groups'][0]['lr']
    epoch_finished_time = datetime.timedelta(seconds=round((cfg['global']['num_epochs'] - epoch) * _time))
    info = {'info': ['Model: {}'.format(cfg['model_tag']),
                     'Train Epoch (C): {}({:.0f}%)'.format(epoch, 100.),
                     'Learning rate: {:.6f}'.format(lr),
                     'Epoch Finished Time: {}'.format(epoch_finished_time)]}
    logger.append(info, 'train', mean=False)
    print(logger.write('train', metric.metric_name['train']))
    logger.safe(False)
    return


def train_user(user, metric, logger, epoch):
    logger.safe(True)
    num_active_users = int(np.ceil(cfg['active_rate'] * cfg['num_users']))
    user_id = torch.arange(cfg['num_users'])[torch.randperm(cfg['num_users'])[:num_active_users]].tolist()
    num_active_users = len(user_id)
    start_time = time.time()
    for i in range(num_active_users):
        dataset = user[user_id[i]].make_dataset()
        if dataset is None:
            continue
        user[user_id[i]].active = True
        user[user_id[i]].train(dataset, metric, logger)
        if i % int((num_active_users * cfg['log_interval']) + 1) == 0:
            _time = (time.time() - start_time) / (i + 1)
            lr = user[user_id[i]].optimizer_state_dict['param_groups'][0]['lr']
            epoch_finished_time = datetime.timedelta(seconds=_time * (num_active_users - i - 1))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['global']['num_epochs'] - epoch) * _time * num_active_users))
            exp_progress = 100. * i / num_active_users
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch (U): {}({:.0f}%)'.format(epoch, exp_progress),
                             'Learning rate: {:.6f}'.format(lr),
                             'ID: {}({}/{})'.format(user_id[i], i + 1, num_active_users),
                             'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']))
    logger.safe(False)
    return


def test(data_loader, model, metric, logger, epoch):
    logger.safe(True)
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
    logger.safe(False)
    return


if __name__ == "__main__":
    main()
