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
    make_batchnorm_dataset_cu
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
    center_dataset['train'], user_dataset['train'], data_separate = separate_dataset_cu(center_dataset['train'],
                                                                                        user_dataset['train'])
    batchnorm_dataset = make_batchnorm_dataset_cu(center_dataset['train'], user_dataset['train'])
    center = make_center(center_dataset)
    data_split, target_split = split_dataset(user_dataset, cfg['num_users'], cfg['data_split_mode'])
    if cfg['resume_mode'] == 1:
        result = resume(cfg['model_tag'])
        last_epoch = result['epoch']
        if last_epoch > 1:
            center = result['center']
            data_split = result['data_split']
            target_split = result['target_split']
            logger = result['logger']
        else:
            logger = make_logger('output/runs/train_{}'.format(cfg['model_tag']))
    else:
        last_epoch = 1
        logger = make_logger('output/runs/train_{}'.format(cfg['model_tag']))
    metric = Metric({'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']})
    for epoch in range(last_epoch, cfg['global']['num_epochs'] + 1):
        logger.safe(True)
        train_center(center, metric, logger)
        user = make_user(user_dataset, data_split, center)
        train_user(user, metric, logger, epoch)
        center.update(user)
        center.make_batchnorm_stats(batchnorm_dataset)
        test(center_dataset['test'], center, metric, logger, epoch)
        logger.safe(False)
        result = {'cfg': cfg, 'epoch': epoch + 1, 'data_split': data_split, 'target_split': target_split,
                  'center': center, 'logger': logger}
        save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    logger.safe(False)
    return


def make_center(center_dataset):
    center = Center(center_dataset, cfg['model_name'], cfg['user_model_name'])
    return center


def train_center(center, metric, logger):
    dataset = center.make_dataset()
    center.train(dataset, metric, logger)
    return


def make_user(user_dataset, data_split, center):
    num_active_users = int(np.ceil(cfg['active_rate'] * cfg['num_users']))
    user_id = torch.arange(cfg['num_users'])[torch.randperm(cfg['num_users'])[:num_active_users]].tolist()
    user = [None for _ in range(num_active_users)]
    for m in range(num_active_users):
        user_dataset_m = {'train': separate_dataset(user_dataset['train'], data_split['train'][m])[0],
                          'test': separate_dataset(user_dataset['test'], data_split['test'][m])[0]}
        user[m] = User(user_id[m], user_dataset_m, cfg['threshold'], cfg['model_name'], cfg['user_model_name'])
        center.distribute(user[m])
    return user


def train_user(user, metric, logger, epoch):
    num_active_users = len(user)
    start_time = time.time()
    for m in range(num_active_users):
        dataset = user[m].make_dataset()
        if dataset is not None:
            user[m].train(dataset, metric, logger)
        if m % int((num_active_users * cfg['log_interval']) + 1) == 0:
            _time = (time.time() - start_time) / (m + 1)
            epoch_finished_time = datetime.timedelta(seconds=_time * (num_active_users - m - 1))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['global']['num_epochs'] - epoch) * _time * num_active_users))
            exp_progress = 100. * m / num_active_users
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'User Train Epoch: {}({:.0f}%)'.format(epoch, exp_progress),
                             'ID: {}({}/{})'.format(user[m].user_id, m + 1, num_active_users),
                             'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']))
    return


def test(dataset, center, metric, logger, epoch):
    center.test(dataset, metric, logger, epoch)
    return


if __name__ == "__main__":
    main()
