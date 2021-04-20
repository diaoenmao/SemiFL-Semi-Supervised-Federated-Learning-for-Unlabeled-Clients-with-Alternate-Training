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
from data import make_data_loader, make_dataset_normal
from utils import to_device, make_optimizer, make_scheduler, collate
from metrics import Accuracy


class Center:
    def __init__(self, center_dataset, teacher_model, model):
        self.center_dataset = center_dataset
        self.teacher_model = teacher_model
        self.model_state_dict = model.state_dict()
        optimizer = make_optimizer(model, 'center')
        scheduler = make_scheduler(optimizer, 'global')
        self.optimizer_state_dict = optimizer.state_dict()
        self.scheduler_state_dict = scheduler.state_dict()

    def distribute(self, user):
        for m in range(len(user)):
            user[m].model_state_dict = copy.deepcopy(self.model_state_dict)
        return

    def update(self, user):
        with torch.no_grad():
            valid_user = [user[i] for i in range(len(user)) if user[i].active and user[i].valid_data]
            num_data = [1 for _ in range(len(valid_user) + 1)]
            weight = torch.tensor(num_data).float().softmax(dim=-1)
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict)
            for k, v in model.named_parameters():
                parameter_type = k.split('.')[-1]
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    tmp_v = weight[-1] * copy.deepcopy(v)
                    for m in range(len(valid_user)):
                        tmp_v += weight[m] * user[m].model_state_dict[k]
                    v.data = tmp_v.data
            self.model_state_dict = model.state_dict()
            for i in range(len(user)):
                user[i].active = False
        return

    def train(self, metric, logger):
        data_loader = make_data_loader(self.center_dataset, 'center')['train']
        model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        model.load_state_dict(self.model_state_dict)
        optimizer = make_optimizer(model, 'center')
        optimizer.load_state_dict(self.optimizer_state_dict)
        scheduler = make_scheduler(optimizer, 'global')
        scheduler.load_state_dict(self.scheduler_state_dict)
        model.train(True)
        for epoch in range(1, cfg['user']['num_epochs'] + 1):
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
        self.model_state_dict = model.state_dict()
        self.optimizer_state_dict = optimizer.state_dict()
        self.scheduler_state_dict = scheduler.state_dict()
        return


class User:
    def __init__(self, user_id, user_dataset, teacher_model, model, threshold):
        self.user_id = user_id
        self.user_dataset = user_dataset
        self.teacher_model = teacher_model
        self.model_state_dict = model.state_dict()
        optimizer = make_optimizer(model, 'user')
        scheduler = make_scheduler(optimizer, 'global')
        self.optimizer_state_dict = optimizer.state_dict()
        self.scheduler_state_dict = scheduler.state_dict()
        self.threshold = threshold
        self.user_dataset['train'], self.mask, self.weight = self.make_dataset(self.user_dataset['train'])
        self.valid_data = False if self.user_dataset['train'] is None else True
        self.active = False

    def make_hard_pseudo_label(self, logits):
        soft_pseudo_label = F.softmax(logits, dim=-1)
        max_p, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
        mask = max_p.ge(self.threshold)
        return hard_pseudo_label, mask

    def make_dataset(self, dataset):
        with torch.no_grad():
            dataset, _transform = make_dataset_normal(dataset)
            data_loader = make_data_loader({'train': dataset}, 'user', shuffle={'train': False})['train']
            self.teacher_model.train(False)
            output = []
            target = []
            for i, input in enumerate(data_loader):
                input = collate(input)
                input = to_device(input, cfg['device'])
                _output = self.teacher_model(input)
                output_i = _output['target']
                target_i = input['target']
                output.append(output_i.cpu())
                target.append(target_i.cpu())
            dataset.transform = _transform
            output = torch.cat(output, dim=0)
            target = torch.cat(target, dim=0)
            acc = Accuracy(output, target)
            new_target, mask = self.make_hard_pseudo_label(output)
            if torch.all(~mask):
                return None, mask, None
            else:
                new_acc = Accuracy(output[mask], target[mask])
                num_labeled = int(mask.float().sum())
                print('Accuracy: {:.3f} ({:.3f}), Number of Labeled: {}'.format(acc, new_acc, num_labeled))
                dataset = copy.deepcopy(dataset)
                dataset.target = new_target.tolist()
                mask = mask.tolist()
                dataset.data = list(compress(dataset.data, mask))
                dataset.target = list(compress(dataset.target, mask))
                dataset.other = {'id': list(range(len(dataset.data)))}
                cls_indx, cls_counts = torch.unique(new_target, return_counts=True)
                num_samples_per_cls = torch.zeros(cfg['target_size'], dtype=torch.float32)
                num_samples_per_cls[cls_indx] = cls_counts.float()
                beta = torch.tensor(0.999, dtype=torch.float32)
                effective_num = 1.0 - beta.pow(num_samples_per_cls)
                weight = (1.0 - beta) / effective_num
                weight[torch.isinf(weight)] = 0
                weight = weight / torch.sum(weight) * (weight > 0).float().sum()
        return dataset, mask, weight

    def train(self, metric, logger):
        data_loader = make_data_loader(self.user_dataset, 'user')['train']
        model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        model.load_state_dict(self.model_state_dict)
        optimizer = make_optimizer(model, 'user')
        optimizer.load_state_dict(self.optimizer_state_dict)
        scheduler = make_scheduler(optimizer, 'global')
        scheduler.load_state_dict(self.scheduler_state_dict)
        model.train(True)
        for epoch in range(1, cfg['user']['num_epochs'] + 1):
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
        scheduler.step()
        self.model_state_dict = model.state_dict()
        self.optimizer_state_dict = optimizer.state_dict()
        self.scheduler_state_dict = scheduler.state_dict()
        return
