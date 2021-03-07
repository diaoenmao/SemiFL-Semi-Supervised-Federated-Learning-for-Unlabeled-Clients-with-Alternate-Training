import torch
import numpy as np
from config import cfg
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


def fetch_dataset(data_name, verbose=True):
    import datasets
    dataset = {}
    if verbose:
        print('fetching data {}...'.format(data_name))
    root = './data/{}'.format(data_name)
    if data_name in ['MNIST', 'CIFAR10']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
    else:
        raise ValueError('Not valid dataset name')
    if verbose:
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
