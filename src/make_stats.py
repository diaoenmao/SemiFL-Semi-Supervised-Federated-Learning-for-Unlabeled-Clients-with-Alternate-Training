import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import datasets
from torchvision import transforms
from config import cfg, process_args
from data import fetch_dataset, make_data_loader
from utils import save, process_control, process_dataset, collate, Stats, makedir_exist_ok

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)

stats_path = os.path.join('res', 'stats')
dim = 1

if __name__ == "__main__":
    makedir_exist_ok(stats_path)
    process_control()
    cfg['seed'] = 0
    data_names = ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'SVHN']
    with torch.no_grad():
        for data_name in data_names:
            cfg['data_name'] = data_name
            root = os.path.join('data', cfg['data_name'])
            dataset = fetch_dataset(cfg['data_name'])
            process_dataset(dataset)
            dataset['train'].transform = datasets.Compose([transforms.ToTensor()])
            data_loader = make_data_loader(dataset, 'global')
            stats = Stats(dim=dim)
            for i, input in enumerate(data_loader['train']):
                input = collate(input)
                stats.update(input['data'])
            stats = (stats.mean.tolist(), stats.std.tolist())
            print(cfg['data_name'], stats)
            save(stats, os.path.join(stats_path, '{}.pt'.format(cfg['data_name'])))
