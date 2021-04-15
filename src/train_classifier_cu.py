import argparse
import datetime
import models
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
from config import cfg
from data import fetch_dataset, make_data_loader, separate_dataset_cu, make_batchnorm_dataset_cu, make_stats_batchnorm
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
        center_control_name = '1_1_none_{}_none_none'.format(cfg['num_supervised'])
        center_model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['model_name'], center_control_name]
        cfg['center_model_tag'] = '_'.join([x for x in center_model_tag_list if x])
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
    center_result = resume(cfg['center_model_tag'])
    data_separate = center_result['data_separate']
    center_dataset['train'], user_dataset['train'], data_separate = separate_dataset_cu(center_dataset['train'],
                                                                                        user_dataset['train'],
                                                                                        data_separate)
    batchnorm_dataset = make_batchnorm_dataset_cu(center_dataset['train'], user_dataset['train'])
    center_data_loader = make_data_loader(center_dataset, cfg['model_name'])
    user_data_loader = make_data_loader(user_dataset, cfg['model_name'])
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    optimizer = make_optimizer(model, cfg['model_name'])
    scheduler = make_scheduler(optimizer, cfg['model_name'])
    if cfg['resume_mode'] == 1:
        result = resume(cfg['model_tag'])
        last_epoch = result['epoch']
        if last_epoch > 1:
            model.load_state_dict(result['center_model_state_dict'])
            optimizer.load_state_dict(result['optimizer_state_dict'])
            scheduler.load_state_dict(result['scheduler_state_dict'])
            logger = result['logger']
        else:
            logger = make_logger('output/runs/train_{}'.format(cfg['model_tag']))
    else:
        last_epoch = 1
        logger = make_logger('output/runs/train_{}'.format(cfg['model_tag']))
    metric = Metric({'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']})
    if cfg['world_size'] > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(cfg['world_size'])))
    center = Center(model)
    user = User(model, cfg['threshold'])
    for epoch in range(last_epoch, cfg[cfg['model_name']]['num_epochs'] + 1):
        logger.safe(True)
        train(center_data_loader['train'], center, optimizer, metric, logger, epoch, 'C')
        test_model = make_stats_batchnorm(batchnorm_dataset, model, cfg['model_name'])
        test(center_data_loader['test'], test_model, metric, logger, epoch, 'C')
        logger.safe(False)
        logger.reset()
        logger.safe(True)
        train(user_data_loader['train'], user, optimizer, metric, logger, epoch, 'U')
        test_model = make_stats_batchnorm(batchnorm_dataset, model, cfg['model_name'])
        test(center_data_loader['test'], test_model, metric, logger, epoch, 'U')
        logger.safe(False)
        scheduler.step()
        model_state_dict = test_model.module.state_dict() if cfg['world_size'] > 1 else test_model.state_dict()
        result = {'cfg': cfg, 'epoch': epoch + 1, 'data_separate': data_separate, 'model_state_dict': model_state_dict,
                  'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                  'logger': logger}
        save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    logger.safe(False)
    return


def train(data_loader, model, optimizer, metric, logger, epoch, tag):
    model.train(True)
    start_time = time.time()
    for i, input in enumerate(data_loader):
        input = collate(input)
        input_size = input['data'].size(0)
        input = to_device(input, cfg['device'])
        optimizer.zero_grad()
        output = model(input)
        if output['loss'] is None:
            continue
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
