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



if __name__ == "__main__":
    p = torch.nn.Parameter(torch.ones(5))
    optimizer = torch.optim.SGD([p], lr=1)
    g = torch.ones(5)*0.5
    p.grad = g
    print(p)
    optimizer.step()
    print(p)