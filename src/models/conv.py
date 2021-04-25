import torch
import torch.nn as nn
from config import cfg
from .utils import init_param, make_batchnorm, loss_fn, kld_loss


class Conv(nn.Module):
    def __init__(self, data_shape, hidden_size, target_size):
        super().__init__()
        blocks = [nn.Conv2d(data_shape[0], hidden_size[0], 3, 1, 1),
                  nn.BatchNorm2d(hidden_size[0]),
                  nn.ReLU(inplace=True),
                  nn.MaxPool2d(2)]
        for i in range(len(hidden_size) - 1):
            blocks.extend([nn.Conv2d(hidden_size[i], hidden_size[i + 1], 3, 1, 1),
                           nn.BatchNorm2d(hidden_size[i + 1]),
                           nn.ReLU(inplace=True),
                           nn.MaxPool2d(2)])
        blocks = blocks[:-1]
        blocks.extend([nn.AdaptiveAvgPool2d(1),
                       nn.Flatten(),
                       nn.Linear(hidden_size[-1], target_size)])
        self.blocks = nn.Sequential(*blocks)

    def f(self, x):
        x = self.blocks(x)
        return x

    def forward(self, input):
        output = {}
        output['target'] = self.f(input['data'])
        if 'weight' in input:
            uda_output = self.f(input['uda'])
            output['loss'] = loss_fn(uda_output, input['target'].detach(), input['weight'])
        else:
            output['loss'] = loss_fn(output['target'], input['target'])
        return output


def conv(track=False):
    data_shape = cfg['data_shape']
    hidden_size = cfg['conv']['hidden_size']
    target_size = cfg['target_size']
    model = Conv(data_shape, hidden_size, target_size)
    model.apply(init_param)
    model.apply(lambda m: make_batchnorm(m, momentum=None, track_running_stats=track))
    return model
