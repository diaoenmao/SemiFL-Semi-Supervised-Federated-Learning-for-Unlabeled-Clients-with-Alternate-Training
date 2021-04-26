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