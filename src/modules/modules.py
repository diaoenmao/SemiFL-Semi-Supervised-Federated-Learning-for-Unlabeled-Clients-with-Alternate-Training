import copy
import datetime
import numpy as np
import sys
import time
import torch
import torch.nn.functional as F
import models
from itertools import compress
from config import cfg
from data import make_data_loader, make_batchnorm_stats
from utils import to_device, make_optimizer, make_scheduler, collate


def make_residual(output, target):
    output.requires_grad = True
    loss = models.loss_fn(output, target, reduction='sum')
    loss.backward()
    residual = - copy.deepcopy(output.grad)
    output.detach_()
    return residual


class Center:
    def __init__(self, center_dataset, batchnorm_dataset, center_parameters, center_model_name, user_model_name):
        self.center_dataset = center_dataset
        self.batchnorm_dataset = batchnorm_dataset
        self.center_model_name = center_model_name
        self.user_model_name = user_model_name
        self.center_parameters, self.buffer = self.initialize(center_parameters)
        self.user_parameters = []
        self.user_learning_rate = []

    def initialize(self, parameters):
        with torch.no_grad():
            model = eval('models.{}().to(cfg["device"])'.format(self.center_model_name))
            model.load_state_dict(parameters)
            model = make_batchnorm_stats(self.batchnorm_dataset, model, 'center')
            parameters = model.state_dict()
            data_loader = make_data_loader(self.center_dataset, 'center', shuffle={'train': False, 'test': False})
            buffer = {split: [] for split in self.center_dataset}
            for split in self.center_dataset:
                for i, input in enumerate(data_loader[split]):
                    input = collate(input)
                    input = to_device(input, cfg['device'])
                    buffer[split].append(model(input)['target'].cpu())
                buffer[split] = torch.cat(buffer[split], dim=0)
        return parameters, buffer

    def distribute(self, user):
        for m in range(len(user)):
            user[m].center_parameters = copy.deepcopy(self.center_parameters)
            user[m].user_parameters = copy.deepcopy(self.user_parameters)
            user[m].user_learning_rate = copy.deepcopy(self.user_learning_rate)
        return

    def update(self, user):
        with torch.no_grad():
            model = eval('models.{}()'.format(self.user_model_name))
            for k, v in model.named_parameters():
                parameter_type = k.split('.')[-1]
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
                    for m in range(len(user)):
                        tmp_v += user[m].user_parameters[-1][k]
                    tmp_v = tmp_v / len(user)
                    v.data = tmp_v.data
            model = make_batchnorm_stats(self.batchnorm_dataset, model.to(cfg['device']), 'user')
            self.user_parameters.append(model.to('cpu').state_dict())
        return

    def search(self):
        model = eval('models.{}().to(cfg["device"])'.format(self.user_model_name))
        model.load_state_dict(self.user_parameters[-1])
        data_loader = make_data_loader(self.center_dataset, 'center', shuffle={'train': False, 'test': False})
        output = {split: [] for split in self.center_dataset}
        for split in self.center_dataset:
            for i, input in enumerate(data_loader[split]):
                input = collate(input)
                input = to_device(input, cfg['device'])
                with torch.no_grad():
                    output[split].append(model(input)['target'].cpu())
            output[split] = torch.cat(output[split], dim=0)
            if split == 'train':
                input = {'buffer': self.buffer['train'],
                         'output': output['train'],
                         'target': torch.tensor(self.center_dataset['train'].target)}
                input = to_device(input, cfg['device'])
                ls = models.linesearch().to(cfg['device'])
                ls.train(True)
                optimizer = make_optimizer(ls, 'linesearch')
                for linearsearch_epoch in range(1, cfg['linesearch']['num_epochs'] + 1):
                    def closure():
                        output = ls(input)
                        optimizer.zero_grad()
                        output['loss'].backward()
                        return output['loss']

                    optimizer.step(closure)
                self.user_learning_rate.append(ls.learning_rate.data.cpu())
                print('Learning Rate: {}'.format(self.user_learning_rate[-1].tolist()))
            with torch.no_grad():
                self.buffer[split] = (self.buffer[split] + self.user_learning_rate[-1] * output[split]).detach()
        return

    def test(self, metric, logger, epoch):
        with torch.no_grad():
            input_size = len(self.center_dataset['test'])
            input = {'target': torch.tensor(self.center_dataset['test'].target)}
            output = {'target': self.buffer['test']}
            output['loss'] = models.loss_fn(output['target'], input['target'])
            evaluation = metric.evaluate(metric.metric_name['test'], input, output)
            logger.append(evaluation, 'test', input_size)
            info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
            logger.append(info, 'test', mean=False)
            print(logger.write('test', metric.metric_name['test']))
        return


class User:
    def __init__(self, user_id, user_dataset, threshold, center_model_name, user_model_name):
        self.user_id = user_id
        self.user_dataset = user_dataset
        self.threshold = threshold
        self.center_model_name = center_model_name
        self.user_model_name = user_model_name
        self.center_parameters = None
        self.user_parameters = []
        self.user_learning_rate = []

    def make_hard_pseudo_label(self, logits):
        soft_pseudo_label = F.softmax(logits, dim=-1)
        max_p, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
        mask = max_p.ge(self.threshold)
        return hard_pseudo_label, mask

    def make_dataset(self):
        data_loader = make_data_loader({'train': self.user_dataset['train']}, 'user', shuffle={'train': False})['train']
        if len(self.user_parameters) == 0:
            center_model = eval('models.{}().to(cfg["device"])'.format(self.center_model_name))
            center_model.load_state_dict(self.center_parameters)
            center_model.train(False)
            buffer = []
            for i, input in enumerate(data_loader):
                input = collate(input)
                input = to_device(input, cfg['device'])
                with torch.no_grad():
                    output = center_model(input)
                buffer_i = output['target']
                buffer.append(buffer_i.cpu())
            self.buffer = torch.cat(buffer, dim=0)
        else:
            user_model = eval('models.{}().to(cfg["device"])'.format(self.user_model_name))
            user_model.load_state_dict(self.user_parameters[-1])
            user_model.train(False)
            new = []
            for i, input in enumerate(data_loader):
                input = collate(input)
                input = to_device(input, cfg['device'])
                with torch.no_grad():
                    output = user_model(input)
                new_i = self.user_learning_rate[-1].to(cfg['device']) * output['target']
                new.append(new_i.cpu())
            new = torch.cat(new, dim=0)
            self.buffer = self.buffer + new
        target, mask = self.make_hard_pseudo_label(self.buffer)
        print('Number of labeled data in User {}: {}'.format(self.user_id, int(mask.float().sum())))
        if torch.all(~mask):
            dataset = None
        else:
            dataset = copy.deepcopy(self.user_dataset['train'])
            dataset.target = target.tolist()
            mask = mask.tolist()
            dataset.data = list(compress(dataset.data, mask))
            dataset.target = list(compress(dataset.target, mask))
            dataset.other = {'id': list(range(len(dataset.data))), 'buffer': list(compress(self.buffer.tolist(), mask))}
        return dataset

    def train(self, dataset, metric, logger):
        data_loader = make_data_loader({'train': dataset}, 'user')['train']
        model = eval('models.{}().to(cfg["device"])'.format(self.user_model_name))
        model.train(True)
        optimizer = make_optimizer(model, 'user')
        scheduler = make_scheduler(optimizer, 'user')
        for epoch in range(1, cfg['user']['num_epochs'] + 1):
            start_time = time.time()
            for i, input in enumerate(data_loader):
                input = collate(input)
                input_size = input['data'].size(0)
                input = to_device(input, cfg['device'])
                optimizer.zero_grad()
                output = model(input)
                output['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                evaluation = metric.evaluate(metric.metric_name['train'], input, output)
                logger.append(evaluation, 'train', n=input_size)
            scheduler.step()
            if epoch % int((cfg['user']['num_epochs'] * cfg['log_interval']) + 1) == 0:
                _time = (time.time() - start_time)
                epoch_finished_time = datetime.timedelta(
                    seconds=round((cfg['user']['num_epochs'] - epoch) * _time))
                epoch_progress = 100. * epoch / cfg['user']['num_epochs']
                info = {'info': ['Model: {}'.format(cfg['model_tag']),
                                 'User ({}) Train Epoch: {}({:.0f}%)'.format(self.user_id, epoch, epoch_progress),
                                 'User Finished Time: {}'.format(epoch_finished_time)]}
                logger.append(info, 'train', mean=False)
                print(logger.write('train', metric.metric_name['train']))
        self.user_parameters.append(model.to('cpu').state_dict())
        return

