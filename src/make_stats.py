import os
import torch
import datasets
from torchvision import transforms
from config import cfg
from data import fetch_dataset, make_data_loader
from utils import save, collate, Stats, makedir_exist_ok, process_control, process_dataset

if __name__ == "__main__":
    stats_path = './res/stats'
    dim = 1
    data_names = ['MNIST', 'CIFAR10', 'CIFAR100', 'SVHN', 'STL10']
    process_control()
    cfg['seed'] = 0
    with torch.no_grad():
        for data_name in data_names:
            dataset = fetch_dataset(data_name)
            dataset['train'].transform=datasets.Compose([transforms.ToTensor()])
            process_dataset(dataset)
            data_loader = make_data_loader(dataset, 'global')
            stats = Stats(dim=dim)
            for i, input in enumerate(data_loader['train']):
                input = collate(input)
                stats.update(input['data'])
            stats = (stats.mean.tolist(), stats.std.tolist())
            print(data_name, stats)
            makedir_exist_ok(stats_path)
            save(stats, os.path.join(stats_path, '{}.pt'.format(data_name)))
