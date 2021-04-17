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
from data import make_data_loader
from utils import to_device, make_optimizer, make_scheduler, collate


def make_residual(output, target):
    output.requires_grad = True
    loss = models.loss_fn(output, target, reduction='sum')
    loss.backward()
    residual = - copy.deepcopy(output.grad)
    output.detach_()
    return residual


class Center:
    def __init__(self, center_dataset, center_model_name, user_model_name):
        self.center_dataset = center_dataset
        self.center_model_name = center_model_name
        self.user_model_name = user_model_name
        self.center_parameters = None
        self.user_parameters = None
        self.center_learning_rate = 1
        self.user_learning_rate = 1
        self.init = self.initialize()

    def initialize(self):
        with torch.no_grad():
            train_target = torch.tensor(self.center_dataset['train'].target)
            _, _, counts = torch.unique(train_target, sorted=True, return_inverse=True, return_counts=True)
            init = (counts / counts.sum()).log().view(1, -1)
        return init

    def make_dataset(self):
        if self.user_parameters is None:
            train_target = torch.tensor(self.center_dataset['train'].target)
            base = self.init.repeat(train_target.size(0), 1)
            target = make_residual(base, train_target).tolist()
            dataset = copy.deepcopy(self.center_dataset['train'])
            dataset.target = target
            base = base.tolist()
            dataset.other = {**dataset.other, 'base': base}
        else:
            user_model = eval('models.{}().to(cfg["device"])'.format(self.user_model_name))
            user_model.load_state_dict(self.user_parameters)
            user_model.train(False)
            data_loader = make_data_loader({'train': self.center_dataset['train']}, 'center', shuffle={'train': False})['train']
            base = []
            target = []
            for i, input in enumerate(data_loader):
                input = collate(input)
                input = to_device(input, cfg['device'])
                with torch.no_grad():
                    user_output = user_model(input)
                base_i = self.init.to(cfg['device']) + self.user_learning_rate * user_output['target']
                target_i = make_residual(base_i, input['target'])
                base.append(base_i.cpu())
                target.append(target_i.cpu())
            base = torch.cat(base, dim=0).tolist()
            target = torch.cat(target, dim=0).tolist()
            dataset = copy.deepcopy(self.center_dataset['train'])
            dataset.target = target
            dataset.other = {**dataset.other, 'base': base}
        return dataset

    def train(self, dataset, metric, logger):
        data_loader = make_data_loader({'train': dataset}, 'center')['train']
        model = eval('models.{}().to(cfg["device"])'.format(self.center_model_name))
        if self.center_parameters is not None:
            model.load_state_dict(self.center_parameters)
        model.train(True)
        optimizer = make_optimizer(model, 'center')
        scheduler = make_scheduler(optimizer, 'center')
        for epoch in range(1, cfg['center']['num_epochs'] + 1):
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
            if epoch % int((cfg['center']['num_epochs'] * cfg['log_interval']) + 1) == 0:
                _time = (time.time() - start_time)
                epoch_finished_time = datetime.timedelta(
                    seconds=round((cfg['center']['num_epochs'] - epoch) * _time))
                epoch_progress = 100. * epoch / cfg['center']['num_epochs']
                info = {'info': ['Model: {}'.format(cfg['model_tag']),
                                 'Center Train Epoch: {}({:.0f}%)'.format(epoch, epoch_progress),
                                 'Center Finished Time: {}'.format(epoch_finished_time)]}
                logger.append(info, 'train', mean=False)
                print(logger.write('train', metric.metric_name['train']))
        self.center_parameters = model.to('cpu').state_dict()
        return

    def make_batchnorm_stats(self, dataset):
        pass
        return

    def test(self, dataset, metric, logger, epoch):
        with torch.no_grad():
            center_model = eval('models.{}().to(cfg["device"])'.format(self.center_model_name))
            center_model.load_state_dict(self.center_parameters)
            center_model.train(False)
            if self.user_parameters is not None:
                user_model = eval('models.{}().to(cfg["device"])'.format(self.user_model_name))
                user_model.load_state_dict(self.user_parameters)
                user_model.train(False)
            data_loader = make_data_loader({'test': dataset}, 'center')['test']
            for i, input in enumerate(data_loader):
                input = collate(input)
                input_size = input['data'].size(0)
                input = to_device(input, cfg['device'])
                center_output = center_model(input)
                output = {}
                if self.user_parameters is not None:
                    user_output = user_model(input)
                    output['target'] = self.init.to(cfg['device']) + self.center_learning_rate * \
                                       center_output['target'] + self.user_learning_rate * user_output['target']
                else:
                    output['target'] = self.init.to(cfg['device']) + self.center_learning_rate * center_output['target']
                output['loss'] = models.loss_fn(output['target'], input['target'])
                evaluation = metric.evaluate(metric.metric_name['test'], input, output)
                logger.append(evaluation, 'test', input_size)
            info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
            logger.append(info, 'test', mean=False)
            print(logger.write('test', metric.metric_name['test']))
        return

    def distribute(self, user):
        user.center_parameters = copy.deepcopy(self.center_parameters)
        user.user_parameters = copy.deepcopy(self.user_parameters)
        user.center_learning_rate = self.center_learning_rate
        user.user_learning_rate = self.user_learning_rate
        user.init = self.init
        return

    def update(self, user):
        valid_user = [user[i] for i in range(len(user)) if user[i].user_parameters is not None]
        if len(valid_user) == 0:
            return
        model = eval('models.{}()'.format(self.user_model_name))
        for k, v in model.named_parameters():
            parameter_type = k.split('.')[-1]
            if 'weight' in parameter_type or 'bias' in parameter_type:
                tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
                for m in range(len(valid_user)):
                    tmp_v += user[m].user_parameters[k]
                tmp_v = tmp_v / len(user)
                v.data = tmp_v.data
        self.user_parameters = model.to('cpu').state_dict()
        return


class User:
    def __init__(self, user_id, user_dataset, threshold, center_model_name, user_model_name):
        self.user_id = user_id
        self.user_dataset = user_dataset
        self.threshold = threshold
        self.center_model_name = center_model_name
        self.user_model_name = user_model_name
        self.center_parameters = None
        self.user_parameters = None
        self.center_learning_rate = 1
        self.user_learning_rate = 1
        self.init = None

    def make_hard_pseudo_label(self, logits):
        soft_pseudo_label = F.softmax(logits, dim=-1)
        max_p, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
        mask = max_p.ge(self.threshold)
        return hard_pseudo_label, mask

    def make_dataset(self):
        center_model = eval('models.{}().to(cfg["device"])'.format(self.center_model_name))
        center_model.load_state_dict(self.center_parameters)
        center_model.train(False)
        # if self.user_parameters is not None:
        #     user_model = eval('models.{}().to(cfg["device"])'.format(self.user_model_name))
        #     user_model.load_state_dict(self.user_parameters)
        #     user_model.train(False)
        data_loader = make_data_loader({'train': self.user_dataset['train']}, 'user', shuffle={'train': False})['train']
        base = []
        target = []
        mask = []
        for i, input in enumerate(data_loader):
            input = collate(input)
            input = to_device(input, cfg['device'])
            # if self.user_parameters is not None:
            #     with torch.no_grad():
            #         center_output = center_model(input)
            #         user_output = user_model(input)
            #     base_i = self.init.to(cfg['device']) + self.center_learning_rate * center_output['target']
            #     logits_i = base_i + self.user_learning_rate * user_output['target']
            # else:
            with torch.no_grad():
                center_output = center_model(input)
            base_i = self.init.to(cfg['device']) + self.center_learning_rate * center_output['target']
            logits_i = base_i
            target_i, mask_i = self.make_hard_pseudo_label(logits_i)
            base.append(base_i.cpu())
            target.append(target_i.cpu())
            mask.append(mask_i.cpu())
        base = torch.cat(base, dim=0).tolist()
        target = torch.cat(target, dim=0)
        mask = torch.cat(mask, dim=0)
        print('Number of labeled data in User {}: {}'.format(self.user_id, int(mask.float().sum())))
        if torch.all(~mask):
            dataset = None
        else:
            dataset = copy.deepcopy(self.user_dataset['train'])
            dataset.target = target.tolist()
            mask = mask.tolist()
            dataset.data = list(compress(dataset.data, mask))
            dataset.target = list(compress(dataset.target, mask))
            dataset.other = {'id': list(range(len(dataset.data))), 'base': list(compress(base, mask))}
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
        self.user_parameters = model.to('cpu').state_dict()
        return
