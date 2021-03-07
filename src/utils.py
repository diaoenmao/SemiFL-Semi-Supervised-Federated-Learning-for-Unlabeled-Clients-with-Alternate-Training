import collections.abc as container_abcs
import errno
import numpy as np
import os
import torch
import torch.optim as optim
from itertools import repeat
from torchvision.utils import save_image
from config import cfg


def check_exists(path):
    return os.path.exists(path)


def makedir_exist_ok(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    return


def save(input, path, protocol=2, mode='torch'):
    dirname = os.path.dirname(path)
    makedir_exist_ok(dirname)
    if mode == 'torch':
        torch.save(input, path, pickle_protocol=protocol)
    elif mode == 'numpy':
        np.save(path, input, allow_pickle=True)
    else:
        raise ValueError('Not valid save mode')
    return


def load(path, mode='torch'):
    if mode == 'torch':
        return torch.load(path, map_location=lambda storage, loc: storage)
    elif mode == 'numpy':
        return np.load(path, allow_pickle=True)
    else:
        raise ValueError('Not valid save mode')
    return


def save_img(img, path, nrow=10, padding=2, pad_value=0, range=None):
    makedir_exist_ok(os.path.dirname(path))
    normalize = False if range is None else True
    save_image(img, path, nrow=nrow, padding=padding, pad_value=pad_value, normalize=normalize, range=range)
    return


def to_device(input, device):
    output = recur(lambda x, y: x.to(y), input, device)
    return output


def ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


def apply_fn(module, fn):
    for n, m in module.named_children():
        if hasattr(m, fn):
            exec('m.{0}()'.format(fn))
        if sum(1 for _ in m.named_children()) != 0:
            exec('apply_fn(m,\'{0}\')'.format(fn))
    return


def recur(fn, input, *args):
    if isinstance(input, torch.Tensor) or isinstance(input, np.ndarray):
        output = fn(input, *args)
    elif isinstance(input, list):
        output = []
        for i in range(len(input)):
            output.append(recur(fn, input[i], *args))
    elif isinstance(input, tuple):
        output = []
        for i in range(len(input)):
            output.append(recur(fn, input[i], *args))
        output = tuple(output)
    elif isinstance(input, dict):
        output = {}
        for key in input:
            output[key] = recur(fn, input[key], *args)
    elif input is None:
        output = None
    else:
        raise ValueError('Not valid input type')
    return output


def process_dataset(dataset):
    cfg['target_size'] = dataset['train'].target_size
    cfg['data_size'] = {split: len(dataset[split]) for split in dataset}
    return


def process_control():
    data_shape = {'MNIST': [1, 28, 28], 'CIFAR10': [3, 32, 32]}
    cfg['data_shape'] = data_shape[cfg['data_name']]
    cfg['conv'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['resnet18'] = {'hidden_size': [64, 128, 256, 512]}
    if 'num_users' in cfg['control']:
        cfg['num_users'] = int(cfg['control']['num_users'])
        cfg['frac'] = float(cfg['control']['frac'])
        cfg['data_split_mode'] = cfg['control']['data_split_mode']
        if cfg['data_name'] in ['MNIST']:
            cfg['optimizer_name'] = 'SGD'
            cfg['lr'] = 1e-2
            cfg['momentum'] = 0.9
            cfg['weight_decay'] = 5e-4
            cfg['scheduler_name'] = 'MultiStepLR'
            cfg['factor'] = 0.1
            if cfg['data_split_mode'] == 'iid':
                cfg['num_epochs'] = {'global': 200, 'local': 5}
                cfg['batch_size'] = {'train': 10, 'test': 50}
                cfg['milestones'] = [100]
            elif 'non-iid' in cfg['data_split_mode']:
                cfg['num_epochs'] = {'global': 400, 'local': 5}
                cfg['batch_size'] = {'train': 10, 'test': 50}
                cfg['milestones'] = [200]
            else:
                raise ValueError('Not valid data_split_mode')
        elif cfg['data_name'] in ['CIFAR10']:
            cfg['optimizer_name'] = 'SGD'
            cfg['lr'] = 1e-1
            cfg['momentum'] = 0.9
            cfg['weight_decay'] = 5e-4
            cfg['scheduler_name'] = 'MultiStepLR'
            cfg['factor'] = 0.1
            if cfg['data_split_mode'] == 'iid':
                cfg['num_epochs'] = {'global': 400, 'local': 5}
                cfg['batch_size'] = {'train': 10, 'test': 50}
                cfg['milestones'] = [150, 250]
            elif 'non-iid' in cfg['data_split_mode']:
                cfg['num_epochs'] = {'global': 800, 'local': 5}
                cfg['batch_size'] = {'train': 10, 'test': 50}
                cfg['milestones'] = [300, 500]
            else:
                raise ValueError('Not valid data_split_mode')
        else:
            raise ValueError('Not valid dataset')
    else:
        for model_name in ['conv', 'resnet18']:
            cfg[model_name]['shuffle'] = {'train': True, 'test': False}
            cfg[model_name]['optimizer_name'] = 'SGD'
            cfg[model_name]['momentum'] = 0.9
            cfg[model_name]['weight_decay'] = 5e-4
            if model_name in ['linear', 'mlp']:
                cfg[model_name]['batch_size'] = {'train': 1024, 'test': 1024}
                cfg[model_name]['lr'] = 1e-1
                cfg[model_name]['num_epochs'] = 100
                cfg[model_name]['scheduler_name'] = 'MultiStepLR'
                cfg[model_name]['factor'] = 0.1
                cfg[model_name]['milestones'] = [50]
            elif model_name in ['conv']:
                cfg[model_name]['batch_size'] = {'train': 512, 'test': 512}
                cfg[model_name]['lr'] = 1e-1
                cfg[model_name]['num_epochs'] = 200
                cfg[model_name]['scheduler_name'] = 'MultiStepLR'
                cfg[model_name]['factor'] = 0.1
                cfg[model_name]['milestones'] = [50, 100]
            else:
                raise ValueError('Not valid model name')
    cfg['stats'] = make_stats()
    return


def make_stats():
    stats = {}
    stats_path = './res/stats'
    makedir_exist_ok(stats_path)
    filenames = os.listdir(stats_path)
    for filename in filenames:
        stats_name = os.path.splitext(filename)[0]
        stats[stats_name] = load(os.path.join(stats_path, filename))
    return stats


class Stats(object):
    def __init__(self, dim):
        self.dim = dim
        self.n_samples = 0
        self.n_features = None
        self.mean = None
        self.std = None

    def update(self, data):
        data = data.transpose(self.dim, -1).reshape(-1, data.size(self.dim))
        if self.n_samples == 0:
            self.n_samples = data.size(0)
            self.n_features = data.size(1)
            self.mean = data.mean(dim=0)
            self.std = data.std(dim=0)
        else:
            m = float(self.n_samples)
            n = data.size(0)
            new_mean = data.mean(dim=0)
            new_std = 0 if n == 1 else data.std(dim=0)
            old_mean = self.mean
            old_std = self.std
            self.mean = m / (m + n) * old_mean + n / (m + n) * new_mean
            self.std = torch.sqrt(m / (m + n) * old_std ** 2 + n / (m + n) * new_std ** 2 + m * n / (m + n) ** 2 * (
                    old_mean - new_mean) ** 2)
            self.n_samples += n
        return


def make_optimizer(model, tag):
    if cfg[tag]['optimizer_name'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=cfg[tag]['lr'], momentum=cfg[tag]['momentum'],
                              weight_decay=cfg[tag]['weight_decay'])
    elif cfg[tag]['optimizer_name'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg[tag]['lr'], weight_decay=cfg[tag]['weight_decay'])
    elif cfg[tag]['optimizer_name'] == 'LBFGS':
        optimizer = optim.LBFGS(model.parameters(), lr=cfg[tag]['lr'])
    else:
        raise ValueError('Not valid optimizer name')
    return optimizer


def make_scheduler(optimizer, tag):
    if cfg[tag]['scheduler_name'] == 'None':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[65535])
    elif cfg[tag]['scheduler_name'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg[tag]['step_size'], gamma=cfg[tag]['factor'])
    elif cfg[tag]['scheduler_name'] == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg[tag]['milestones'],
                                                   gamma=cfg[tag]['factor'])
    elif cfg[tag]['scheduler_name'] == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif cfg[tag]['scheduler_name'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg[tag]['num_epochs']['global'],
                                                         eta_min=cfg[tag]['min_lr'])
    elif cfg[tag]['scheduler_name'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg[tag]['factor'],
                                                         patience=cfg[tag]['patience'], verbose=False,
                                                         threshold=cfg[tag]['threshold'], threshold_mode='rel',
                                                         min_lr=cfg[tag]['min_lr'])
    elif cfg[tag]['scheduler_name'] == 'CyclicLR':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg[tag]['lr'], max_lr=10 * cfg[tag]['lr'])
    else:
        raise ValueError('Not valid scheduler name')
    return scheduler


def resume(model, model_tag, optimizer=None, scheduler=None, load_tag='checkpoint', verbose=True):
    if os.path.exists('./output/model/{}_{}.pt'.format(model_tag, load_tag)):
        checkpoint = load('./output/model/{}_{}.pt'.format(model_tag, load_tag))
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_dict'])
        logger = checkpoint['logger']
        if verbose:
            print('Resume from {}'.format(last_epoch))
    else:
        print('Not exists model tag: {}, start from scratch'.format(model_tag))
        from datetime import datetime
        from logger import Logger
        last_epoch = 1
        logger_path = 'output/runs/train_{}_{}'.format(cfg['model_tag'], datetime.now().strftime('%b%d_%H-%M-%S'))
        logger = Logger(logger_path)
    return last_epoch, model, optimizer, scheduler, logger


def collate(input):
    for k in input:
        input[k] = torch.stack(input[k], 0)
    return input
