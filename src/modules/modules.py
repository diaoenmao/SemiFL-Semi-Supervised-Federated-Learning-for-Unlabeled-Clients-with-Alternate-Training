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
from metrics import Accuracy


class Server:
    def __init__(self, model):
        self.model_state_dict = copy.deepcopy(model.state_dict())
        optimizer = make_optimizer(model, 'local')
        self.optimizer_state_dict = optimizer.state_dict()
        global_optimizer = make_optimizer(model, 'global')
        self.global_optimizer_state_dict = global_optimizer.state_dict()

    def distribute(self, dataset, client):
        model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        model.load_state_dict(self.model_state_dict)
        model = make_batchnorm_stats(dataset, model, 'global')
        for m in range(len(client)):
            client[m].model_state_dict = copy.deepcopy(model.state_dict())
        return

    def update(self, client):
        with torch.no_grad():
            valid_client = [client[i] for i in range(len(client)) if client[i].active]
            if len(valid_client) > 0:
                model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
                model.load_state_dict(self.model_state_dict)
                global_optimizer = make_optimizer(model, 'global')
                global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                global_optimizer.zero_grad()
                logits = [1 for _ in range(len(valid_client))]
                weight = torch.tensor(logits).float().softmax(dim=-1)
                for k, v in model.named_parameters():
                    parameter_type = k.split('.')[-1]
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        tmp_v = v.data.new_zeros(v.size())
                        for m in range(len(valid_client)):
                            tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                        v.grad = (v.data - tmp_v).detach()
                global_optimizer.step()
                self.global_optimizer_state_dict = global_optimizer.state_dict()
                self.model_state_dict = model.state_dict()
            for i in range(len(client)):
                client[i].active = False
        return

    def train(self, dataset, lr, metric, logger):
        data_loader = make_data_loader({'train': dataset}, 'server')['train']
        model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        model.load_state_dict(self.model_state_dict)
        self.optimizer_state_dict['param_groups'][0]['lr'] = lr
        optimizer = make_optimizer(model, 'local')
        optimizer.load_state_dict(self.optimizer_state_dict)
        model.train(True)
        for epoch in range(1, cfg['local']['num_epochs'] + 1):
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
        with torch.no_grad():
            model_state_dict = model.state_dict()
            _model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            _model.load_state_dict(self.model_state_dict)
            global_optimizer = make_optimizer(_model, 'global')
            global_optimizer.load_state_dict(self.global_optimizer_state_dict)
            global_optimizer.zero_grad()
            for k, v in _model.named_parameters():
                parameter_type = k.split('.')[-1]
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    tmp_v = model_state_dict[k]
                    v.grad = (v.data - tmp_v).detach()
            global_optimizer.step()
            self.global_optimizer_state_dict = global_optimizer.state_dict()
            self.model_state_dict = _model.state_dict()
        return


class Client:
    def __init__(self, client_id, model, data_split, threshold):
        self.client_id = client_id
        self.data_split = data_split
        self.model_state_dict = copy.deepcopy(model.state_dict())
        optimizer = make_optimizer(model, 'local')
        self.optimizer_state_dict = optimizer.state_dict()
        self.threshold = threshold
        self.active = False
        self.buffer = None

    def make_hard_pseudo_label(self, soft_pseudo_label):
        max_p, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
        mask = max_p.ge(self.threshold)
        return hard_pseudo_label, mask

    def make_weight(self, target):
        cls_indx, cls_counts = torch.unique(target, return_counts=True)
        num_samples_per_cls = torch.zeros(cfg['target_size'], dtype=torch.float32)
        num_samples_per_cls[cls_indx] = cls_counts.float()
        beta = torch.tensor(0.999, dtype=torch.float32)
        effective_num = 1.0 - beta.pow(num_samples_per_cls)
        weight = (1.0 - beta) / effective_num
        weight[torch.isinf(weight)] = 0
        weight = weight / torch.sum(weight) * (weight > 0).float().sum()
        return weight

    def make_dataset(self, dataset):
        with torch.no_grad():
            data_loader = make_data_loader({'train': dataset}, 'global', shuffle={'train': False})['train']
            model = eval('models.{}(track=True).to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict)
            model.train(False)
            output = []
            target = []
            for i, input in enumerate(data_loader):
                input = collate(input)
                input = to_device(input, cfg['device'])
                _output = model(input)
                output_i = _output['target']
                target_i = input['target']
                output.append(output_i.cpu())
                target.append(target_i.cpu())
            output = torch.cat(output, dim=0)
            target = torch.cat(target, dim=0)
            self.buffer = F.softmax(output, dim=-1)
            acc = Accuracy(self.buffer, target)
            new_target, mask = self.make_hard_pseudo_label(self.buffer)
            if torch.all(~mask):
                print('Accuracy: {:.3f}, Number of Labeled: 0({})'.format(acc, len(output)))
                return None
            else:
                new_acc = Accuracy(self.buffer[mask], target[mask])
                num_labeled = int(mask.float().sum())
                print('Accuracy: {:.3f} ({:.3f}), Number of Labeled: {}({})'.format(acc, new_acc, num_labeled,
                                                                                    len(output)))
                dataset = copy.deepcopy(dataset)
                dataset.target = new_target.tolist()
                mask = mask.tolist()
                dataset.data = list(compress(dataset.data, mask))
                dataset.target = list(compress(dataset.target, mask))
                dataset.other = {'id': list(range(len(dataset.data)))}
                self.weight = self.make_weight(new_target)
                return dataset

    def train(self, dataset, lr, metric, logger):
        data_loader = make_data_loader({'train': dataset}, 'client')['train']
        model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        model.load_state_dict(self.model_state_dict, strict=False)
        self.optimizer_state_dict['param_groups'][0]['lr'] = lr
        optimizer = make_optimizer(model, 'local')
        optimizer.load_state_dict(self.optimizer_state_dict)
        model.train(True)
        for epoch in range(1, cfg['local']['num_epochs'] + 1):
            for i, input in enumerate(data_loader):
                input = collate(input)
                input_size = input['data'].size(0)
                input['weight'] = self.weight
                input = to_device(input, cfg['device'])
                optimizer.zero_grad()
                output = model(input)
                output['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                evaluation = metric.evaluate(metric.metric_name['train'], input, output)
                logger.append(evaluation, 'train', n=input_size)
        self.model_state_dict = model.state_dict()
        self.optimizer_state_dict = optimizer.state_dict()
        return
