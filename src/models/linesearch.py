import torch
import torch.nn as nn
from config import cfg
from .utils import loss_fn


class LineSearch(nn.Module):
    def __init__(self, target_size):
        super().__init__()
        self.learning_rate = nn.Parameter(torch.zeros(1, target_size))

    def forward(self, input):
        output = {}
        output['loss'] = loss_fn(input['buffer'] + self.learning_rate * input['output'], input['target'])
        return output


def linesearch():
    target_size = cfg['target_size']
    # target_size = 1
    model = LineSearch(target_size)
    return model
