import argparse
import datetime
import models
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
from config import cfg
from data import fetch_dataset, make_data_loader
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
from logger import Logger

# if __name__ == "__main__":
#     p = torch.nn.Parameter(torch.ones(5))
#     optimizer = torch.optim.SGD([p], lr=1)
#     g = torch.ones(5)*0.5
#     p.grad = g
#     print(p)
#     optimizer.step()
#     print(p)


# if __name__ == "__main__":
#     import torch.nn.functional as F
#     m = torch.distributions.dirichlet.Dirichlet(torch.ones(10))
#     x = torch.randn(1, 10)
#     p = F.softmax(x, dim=-1)
#     y = m.sample((1,))
#     label = (y.topk(1, 1, True, True)[1]).view(-1)
#     onehot = F.one_hot(label, 10).float()
#     m = (p + onehot) / 2
#     print(y)
#     print(label)
#     print(onehot)
#     ce = F.cross_entropy(x, label, reduction='mean')
#     kld = F.kl_div(F.log_softmax(x, dim=-1), onehot, reduction='batchmean')
#     js = F.kl_div(F.log_softmax(x, dim=-1), m, reduction='batchmean')
#     js += F.kl_div(F.log_softmax(onehot.exp(), dim=-1), onehot, reduction='batchmean')
#     js /= 2
#     print(ce)
#     print(kld)
#     print(js)

# if __name__ == "__main__":
#     process_control()
#     cfg['target_size'] = 10
#     model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
#     optimizer = make_optimizer(model, 'server')
#     print(optimizer.state_dict())
#     exit()
#     scheduler = make_scheduler(optimizer, 'global')
#     lr = []
#     for i in range(300):
#         lr.append(optimizer.param_groups[0]['lr'])
#         scheduler.step()
#     print(lr)
#     print(lr[285])


# if __name__ == "__main__":
#     import numpy as np
#
#     process_control()
#     # cfg['data_split_mode'] = 'non-iid-d-0.3'
#     cfg['data_split_mode'] = 'non-iid-l-2'
#     dataset = fetch_dataset(cfg['data_name'])
#     process_dataset(dataset)
#     dataset = dataset['train']
#     num_users = 100
#     target_split = None
#     target = torch.tensor(dataset.target)
#     data_split_mode_list = cfg['data_split_mode'].split('-')
#     data_split_mode_tag = data_split_mode_list[-2]
#     if data_split_mode_tag == 'l':
#         data_split = {i: [] for i in range(num_users)}
#         shard_per_user = int(data_split_mode_list[-1])
#         target_idx_split = {}
#         shard_per_class = int(shard_per_user * num_users / cfg['target_size'])
#         for target_i in range(cfg['target_size']):
#             target_idx = torch.where(target == target_i)[0]
#             num_leftover = len(target_idx) % shard_per_class
#             leftover = target_idx[-num_leftover:] if num_leftover > 0 else []
#             new_target_idx = target_idx[:-num_leftover] if num_leftover > 0 else target_idx
#             new_target_idx = new_target_idx.reshape((shard_per_class, -1)).tolist()
#             for i, leftover_target_idx in enumerate(leftover):
#                 new_target_idx[i] = new_target_idx[i] + [leftover_target_idx.item()]
#             target_idx_split[target_i] = new_target_idx
#         target_split = list(range(cfg['target_size'])) * shard_per_class
#         target_split = torch.tensor(target_split)[torch.randperm(len(target_split))].tolist()
#         target_split = torch.tensor(target_split).reshape((num_users, -1)).tolist()
#         for i in range(num_users):
#             for target_i in target_split[i]:
#                 idx = torch.randint(len(target_idx_split[target_i]), (1,)).item()
#                 data_split[i].extend(target_idx_split[target_i].pop(idx))
#     elif data_split_mode_tag == 'd':
#         beta = float(data_split_mode_list[-1])
#         min_size = 0
#         required_min_size = 25
#         N = target.size(0)
#         while min_size < required_min_size:
#             data_split = [[] for _ in range(num_users)]
#             for target_i in range(cfg['target_size']):
#                 target_idx = torch.where(target == target_i)[0]
#                 dir = torch.distributions.dirichlet.Dirichlet(torch.tensor(beta).repeat(num_users))
#                 proportions = dir.sample()
#                 proportions = torch.tensor(
#                     [p * (len(data_split_idx) < (N / num_users)) for p, data_split_idx in zip(proportions, data_split)])
#                 proportions = proportions / proportions.sum()
#                 split_idx = (torch.cumsum(proportions, dim=-1) * len(target_idx)).long().tolist()[:-1]
#                 split_idx = torch.tensor_split(target_idx, split_idx)
#                 data_split = [data_split_idx + idx.tolist() for data_split_idx, idx in zip(data_split, split_idx)]
#             min_size = min([len(data_split_idx) for data_split_idx in data_split])
#         data_split = {i: data_split[i] for i in range(num_users)}
#     else:
#         raise ValueError('Not valid data split mode tag')
#     test = [x for i in data_split for x in data_split[i]]
#     print(len(test))
#     print(np.unique(test).shape)
#     target_split = [target[torch.tensor(data_split[i])] for i in data_split]
#     test = []
#     for i in range(len(target_split)):
#         u, count = np.unique(target_split[i], return_counts=True)
#         test.append({u[k]: count[k] for k in range(len(u))})
#     print(test)
#     exit()
