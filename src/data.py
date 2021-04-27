import copy
import torch
import numpy as np
import models
from config import cfg
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from utils import collate, to_device

data_stats = {'MNIST': ((0.1307,), (0.3081,)), 'CIFAR10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
              'CIFAR100': ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
              'SVHN': ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
              'STL10': ((0.4409, 0.4279, 0.3868), (0.2683, 0.2610, 0.2687))}


def fetch_dataset(data_name):
    import datasets
    dataset = {}
    print('fetching data {}...'.format(data_name))
    root = './data/{}'.format(data_name)
    if data_name in ['MNIST']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['train'].transform = datasets.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset['test'].transform = datasets.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    elif data_name in ['CIFAR10', 'CIFAR100']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['train'].transform = datasets.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset['test'].transform = datasets.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    elif data_name in ['SVHN']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['train'].transform = datasets.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset['test'].transform = datasets.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])

    elif data_name in ['STL10']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['train'].transform = datasets.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(96, padding=12, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset['test'].transform = datasets.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    else:
        raise ValueError('Not valid dataset name')
    print('data ready')
    return dataset


def input_collate(batch):
    if isinstance(batch[0], dict):
        output = {key: [] for key in batch[0].keys()}
        for b in batch:
            for key in b:
                output[key].append(b[key])
        return output
    else:
        return default_collate(batch)


def make_data_loader(dataset, tag, shuffle=None):
    data_loader = {}
    for k in dataset:
        _shuffle = cfg[tag]['shuffle'][k] if shuffle is None else shuffle[k]
        data_loader[k] = DataLoader(dataset=dataset[k], shuffle=_shuffle, batch_size=cfg[tag]['batch_size'][k],
                                    pin_memory=True, num_workers=cfg['num_workers'], collate_fn=input_collate,
                                    worker_init_fn=np.random.seed(cfg['seed']))
    return data_loader


def split_dataset(dataset, num_users, data_split_mode):
    data_split = {}
    if data_split_mode == 'iid':
        data_split['train'], target_split = iid(dataset['train'], num_users)
        data_split['test'], _ = iid(dataset['test'], num_users)
    elif 'non-iid' in cfg['data_split_mode']:
        data_split['train'], target_split = non_iid(dataset['train'], num_users)
        data_split['test'], _ = non_iid(dataset['test'], num_users, target_split)
    else:
        raise ValueError('Not valid data split mode')
    return data_split, target_split


def iid(dataset, num_users):
    target = torch.tensor(dataset.target)
    num_items = int(len(dataset) / num_users)
    data_split, idx = {}, list(range(len(dataset)))
    target_split = {}
    for i in range(num_users):
        num_items_i = min(len(idx), num_items)
        data_split[i] = torch.tensor(idx)[torch.randperm(len(idx))[:num_items_i]].tolist()
        target_split[i] = torch.unique(target[data_split[i]]).tolist()
        idx = list(set(idx) - set(data_split[i]))
    return data_split, target_split


def non_iid(dataset, num_users, target_split=None):
    target = np.array(dataset.target)
    shard_per_user = int(cfg['data_split_mode'].split('-')[-1])
    data_split = {i: [] for i in range(num_users)}
    target_idx_split = {}
    for i in range(len(target)):
        target_i = target[i].item()
        if target_i not in target_idx_split:
            target_idx_split[target_i] = []
        target_idx_split[target_i].append(i)
    shard_per_class = int(shard_per_user * num_users / cfg['target_size'])
    for target_i in target_idx_split:
        target_idx = target_idx_split[target_i]
        num_leftover = len(target_idx) % shard_per_class
        leftover = target_idx[-num_leftover:] if num_leftover > 0 else []
        new_target_idx = np.array(target_idx[:-num_leftover]) if num_leftover > 0 else np.array(target_idx)
        new_target_idx = new_target_idx.reshape((shard_per_class, -1)).tolist()
        for i, leftover_target_idx in enumerate(leftover):
            new_target_idx[i] = np.concatenate([new_target_idx[i], [leftover_target_idx]])
        target_idx_split[target_i] = new_target_idx
    if target_split is None:
        target_split = list(range(cfg['target_size'])) * shard_per_class
        target_split = torch.tensor(target_split)[torch.randperm(len(target_split))].tolist()
        target_split = np.array(target_split).reshape((num_users, -1)).tolist()
        for i in range(len(target_split)):
            target_split[i] = np.unique(target_split[i]).tolist()
    for i in range(num_users):
        for target_i in target_split[i]:
            idx = torch.arange(len(target_idx_split[target_i]))[
                torch.randperm(len(target_idx_split[target_i]))[0]].item()
            data_split[i].extend(target_idx_split[target_i].pop(idx))
    return data_split, target_split


def separate_dataset(dataset, idx):
    separated_dataset = copy.deepcopy(dataset)
    separated_dataset.data = [dataset.data[s] for s in idx]
    separated_dataset.target = [dataset.target[s] for s in idx]
    separated_dataset.other['id'] = list(range(len(separated_dataset.data)))
    return separated_dataset


def separate_dataset_su(server_dataset, client_dataset=None, supervised_idx=None):
    if supervised_idx is None:
        if cfg['data_name'] in ['STL10']:
            supervised_idx = list(range(cfg['num_supervised']))
        else:
            if cfg['num_supervised'] == -1:
                supervised_idx = list(range(len(server_dataset)))
            else:
                target = np.array(server_dataset.target)
                num_supervised_per_class = cfg['num_supervised'] // cfg['target_size']
                supervised_idx = []
                for i in range(cfg['target_size']):
                    idx = np.where(target == i)[0]
                    idx = np.random.choice(idx, num_supervised_per_class, False)
                    supervised_idx.extend(idx)
    if cfg['client_data_name'] == 'none' or cfg['data_name'] == cfg['client_data_name']:
        idx = list(range(len(server_dataset)))
        unsupervised_idx = list(set(idx) - set(supervised_idx))
    else:
        unsupervised_idx = list(range(len(client_dataset)))
    _server_dataset = separate_dataset(server_dataset, supervised_idx)
    if client_dataset is None:
        _client_dataset = separate_dataset(server_dataset, unsupervised_idx)
    else:
        _client_dataset = separate_dataset(client_dataset, unsupervised_idx)
        transform = TransformUDA(*data_stats[cfg['client_data_name']])
        _client_dataset.transform = transform
    return _server_dataset, _client_dataset, supervised_idx


def make_batchnorm_dataset_su(server_dataset, client_dataset):
    batchnorm_dataset = copy.deepcopy(server_dataset)
    if cfg['data_name'] == cfg['client_data_name']:
        batchnorm_dataset.data = batchnorm_dataset.data + client_dataset.data
        batchnorm_dataset.target = batchnorm_dataset.target + client_dataset.target
        batchnorm_dataset.other['id'] = batchnorm_dataset.other['id'] + client_dataset.other['id']
    return batchnorm_dataset


def make_dataset_normal(dataset):
    import datasets
    _transform = dataset.transform
    transform = datasets.Compose([transforms.ToTensor(), transforms.Normalize(*data_stats[cfg['data_name']])])
    dataset.transform = transform
    return dataset, _transform


def make_batchnorm_stats(dataset, model, tag):
    with torch.no_grad():
        test_model = copy.deepcopy(model)
        test_model.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=True))
        dataset, _transform = make_dataset_normal(dataset)
        data_loader = make_data_loader({'train': dataset}, tag, shuffle={'train': False})['train']
        test_model.train(True)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input = to_device(input, cfg['device'])
            test_model(input)
        dataset.transform = _transform
    return test_model


class TransformUDA(object):
    def __init__(self, mean, std):
        import datasets
        if cfg['client_data_name'] in ['CIFAR10', 'CIFAR100']:
            self.weak = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            self.strong = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                datasets.RandAugment(n=2, m=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        elif cfg['client_data_name'] in ['SVHN']:
            self.weak = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            self.strong = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                datasets.RandAugment(n=2, m=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        elif cfg['client_data_name'] in ['STL10']:
            self.weak = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(96, padding=12, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            self.strong = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(96, padding=12, padding_mode='reflect'),
                datasets.RandAugment(n=2, m=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            raise ValueError('Not valid dataset')

    def __call__(self, input):
        data = self.weak(input['data'])
        uda = self.strong(input['data'])
        input = {**input, 'data': data, 'uda': uda}
        return input
