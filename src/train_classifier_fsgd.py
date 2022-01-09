import argparse
import copy
import datetime
import models
import os
import shutil
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from data import fetch_dataset, split_dataset, make_data_loader, separate_dataset_su, make_batchnorm_stats, \
    make_batchnorm_dataset_su
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
from logger import make_logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


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
    server_dataset = fetch_dataset(cfg['data_name'])
    client_dataset = fetch_dataset(cfg['data_name'])
    process_dataset(server_dataset)
    server_dataset['train'], client_dataset['train'], supervised_idx = separate_dataset_su(server_dataset['train'],
                                                                                           client_dataset['train'])
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    optimizer = make_optimizer(model, 'local')
    scheduler = make_scheduler(optimizer, 'global')
    batchnorm_dataset = make_batchnorm_dataset_su(server_dataset['train'], client_dataset['train'])
    data_split = split_dataset(client_dataset, cfg['num_clients'], cfg['data_split_mode'])
    metric = Metric({'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']})
    if cfg['resume_mode'] == 1:
        result = resume(cfg['model_tag'])
        last_epoch = result['epoch']
        if last_epoch > 1:
            data_split = result['data_split']
            supervised_idx = result['supervised_idx']
            model.load_state_dict(result['model_state_dict'])
            optimizer.load_state_dict(result['optimizer_state_dict'])
            scheduler.load_state_dict(result['scheduler_state_dict'])
            logger = result['logger']
        else:
            logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    else:
        last_epoch = 1
        logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    if cfg['world_size'] > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(cfg['world_size'])))
    for epoch in range(last_epoch, cfg['global']['num_epochs'] + 1):
        client_sampler = ClientSampler(cfg['client']['batch_size']['train'], cfg['active_rate'], data_split['train'])
        server_sampler = ServerSampler(len(client_sampler), cfg['server']['batch_size']['train'],
                                       len(server_dataset['train']))
        server_dataloader = make_data_loader(server_dataset, 'server',
                                             batch_sampler={'train': server_sampler, 'test': None})
        client_dataloader = make_data_loader(client_dataset, 'client',
                                             batch_sampler={'train': client_sampler, 'test': None})
        logger.safe(True)
        train(server_dataloader['train'], client_dataloader['train'], model, optimizer, metric, logger, epoch)
        test_model = make_batchnorm_stats(batchnorm_dataset, model, 'global')
        test(server_dataloader['test'], test_model, metric, logger, epoch)
        scheduler.step()
        logger.safe(False)
        model_state_dict = model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict()
        result = {'cfg': cfg, 'epoch': epoch + 1, 'model_state_dict': model_state_dict,
                  'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                  'supervised_idx': supervised_idx, 'data_split': data_split, 'logger': logger}
        save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    logger.safe(False)
    return


def train(server_dataloader, client_dataloader, model, optimizer, metric, logger, epoch):
    model.train(True)
    start_time = time.time()
    for i, (server_input, client_input) in enumerate(zip(server_dataloader, client_dataloader)):
        server_input = collate(server_input)
        client_input = collate(client_input)
        server_input = to_device(server_input, cfg['device'])
        client_input = to_device(client_input, cfg['device'])
        with torch.no_grad():
            model.train(False)
            client_output_ = model(client_input)
            buffer = torch.softmax(client_output_['target'], dim=-1)
            new_target, mask = make_hard_pseudo_label(buffer)
            client_input['target'] = new_target.detach()
        input_size = server_input['data'].size(0)
        optimizer.zero_grad()
        output = model(server_input)
        output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
        if torch.any(mask):
            client_input['loss_mode'] = 'fix'
            output_ = model(client_input)
            output['loss'] += output_['loss']
        output['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        evaluation = metric.evaluate(metric.metric_name['train'], server_input, output)
        logger.append(evaluation, 'train', n=input_size)
        if i % int((len(server_dataloader) * cfg['log_interval']) + 1) == 0:
            _time = (time.time() - start_time) / (i + 1)
            lr = optimizer.param_groups[0]['lr']
            epoch_finished_time = datetime.timedelta(seconds=round(_time * (len(server_dataloader) - i - 1)))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['global']['num_epochs'] - epoch) * _time * len(server_dataloader)))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * i / len(server_dataloader)),
                             'Learning rate: {:.6f}'.format(lr), 'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']))
    return


def test(data_loader, model, metric, logger, epoch):
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


class ServerSampler(torch.utils.data.Sampler):
    def __init__(self, num_batches, batch_size, data_size):
        total_size = num_batches * batch_size
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.data_size = data_size
        self.idx = []
        for i in range(int(np.ceil(total_size / data_size))):
            idx_i = torch.randperm(data_size)
            self.idx.append(idx_i)
        self.idx = torch.cat(self.idx, dim=0)[:total_size]
        self.idx = torch.chunk(self.idx, self.num_batches)

    def __iter__(self):
        yield from self.idx

    def __len__(self):
        return len(self.idx)


class ClientSampler(torch.utils.data.Sampler):
    def __init__(self, batch_size, active_rate, data_split):
        self.batch_size = batch_size
        self.active_rate = active_rate
        self.data_split = data_split
        self.num_active_clients = int(self.active_rate * len(self.data_split))
        self.reset()

    def reset(self):
        self.data_split_ = copy.deepcopy(self.data_split)
        self.client_idx = torch.arange(len(self.data_split))
        self.idx = []
        while len(self.client_idx) > 0:
            num_active_clients = min(self.num_active_clients, len(self.client_idx))
            active_client_idx = self.client_idx[torch.randperm(len(self.client_idx))][:num_active_clients]
            batch_idx = []
            for i in range(len(active_client_idx)):
                data_split_i = self.data_split_[active_client_idx[i].item()]
                batch_size_i = min(self.batch_size, len(data_split_i))
                batch_idx.extend(data_split_i[:batch_size_i])
                self.data_split_[active_client_idx[i].item()] = self.data_split_[active_client_idx[i].item()][
                                                                batch_size_i:]
            self.client_idx = torch.tensor([i for i in range(len(self.data_split_)) if len(self.data_split_[i]) > 0])
            self.idx.append(batch_idx)
        return

    def __iter__(self):
        yield from self.idx

    def __len__(self):
        return len(self.idx)


def make_hard_pseudo_label(soft_pseudo_label):
    max_p, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
    mask = max_p.ge(cfg['threshold'])
    return hard_pseudo_label, mask


if __name__ == "__main__":
    main()
