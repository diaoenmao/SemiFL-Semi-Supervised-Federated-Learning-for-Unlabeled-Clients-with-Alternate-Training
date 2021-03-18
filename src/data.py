import copy
import torch
import numpy as np
import models
from config import cfg
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.utils.data.dataloader import default_collate
from utils import collate, to_device


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
            transforms.Normalize((0.1307,), (0.3081,))])
        dataset['test'].transform = datasets.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
    elif data_name in ['CIFAR10', 'CIFAR100']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['train'].transform = datasets.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        dataset['test'].transform = datasets.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        if 'ra' in cfg['augment']:
            dataset['train'].transform.transforms.insert(0, datasets.RandAugment(3, 5))
            dataset['train'].transform.transforms.append(datasets.CutoutDefault(16))
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
    cfg['non-iid-n'] = int(cfg['data_split_mode'].split('-')[-1])
    shard_per_user = cfg['non-iid-n']
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


def separate_dataset(teacher_dataset, student_dataset, supervise_rate):
    num_items = int(len(teacher_dataset) * supervise_rate)
    idx = list(range(len(teacher_dataset)))
    randperm = torch.randperm(len(idx))
    data_supervise = torch.tensor(idx)[randperm[:num_items]]
    teacher_dataset = Subset(teacher_dataset, data_supervise)
    if cfg['data_name'] == cfg['student_data_name']:
        data_unsupervise = torch.tensor(idx)[randperm[num_items:]]
        student_dataset = Subset(student_dataset, data_unsupervise)
    else:
        idx = list(range(len(student_dataset)))
        data_unsupervise = torch.tensor(idx)
        student_dataset = Subset(student_dataset, data_unsupervise)
    return teacher_dataset, student_dataset


def make_stats_batchnorm(dataset, model, tag):
    import datasets
    with torch.no_grad():
        test_model = copy.deepcopy(model)
        test_model.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=True))
        _transform = dataset.transform
        if cfg['data_name'] in ['MNIST']:
            dataset.transform = datasets.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.1307,), (0.3081,))])
        elif cfg['data_name'] in ['CIFAR10', 'CIFAR100']:
            dataset.transform = datasets.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                       (0.2023, 0.1994, 0.2010))])
        else:
            raise ValueError('Not valid dataset name')
        data_loader = make_data_loader({'train': dataset}, tag, shuffle={'train': False})['train']
        test_model.train(True)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input = to_device(input, cfg['device'])
            test_model(input)
        dataset.transform = _transform
    return test_model


def make_stats_batchnorm_fed(teacher_dataset, student_dataset, model, mode, tag):
    import datasets
    with torch.no_grad():
        test_model = copy.deepcopy(model)
        test_model.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=True))
        teacher_transform = teacher_dataset.dataset.transform
        student_transform = student_dataset.dataset.transform
        if cfg['data_name'] in ['MNIST']:
            transform = datasets.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))])
        elif cfg['data_name'] in ['CIFAR10', 'CIFAR100']:
            transform = datasets.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                               (0.2023, 0.1994, 0.2010))])
        else:
            raise ValueError('Not valid dataset name')
        teacher_dataset.dataset.transform = transform
        student_dataset.dataset.transform = transform
        if mode == 'a':
            if cfg['data_name'] == cfg['student_data_name']:
                dataset = ConcatDataset([teacher_dataset, student_dataset])
            else:
                dataset = ConcatDataset([student_dataset])
        elif mode == 'b':
            if cfg['data_name'] == cfg['student_data_name']:
                dataset = ConcatDataset([teacher_dataset, student_dataset])
            else:
                dataset = ConcatDataset([teacher_dataset])
        else:
            raise ValueError('Not valid mode')
        data_loader = make_data_loader({'train': dataset}, tag, shuffle={'train': False})['train']
        test_model.train(True)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input = to_device(input, cfg['device'])
            test_model(input)
        teacher_dataset.dataset.transform = teacher_transform
        student_dataset.dataset.transform = student_transform
    return test_model
