import torch
import torch.nn as nn
from config import cfg
from .utils import init_param, make_batchnorm, loss_fn, kld_loss


class Conv(nn.Module):
    def __init__(self, data_shape, hidden_size, target_size):
        super().__init__()
        blocks = [nn.Conv2d(data_shape[0], hidden_size[0], 5, 1, 2),
                  nn.ReLU(inplace=True),
                  nn.MaxPool2d(2),
                  nn.Conv2d(hidden_size[0], hidden_size[1], 5, 1, 2),
                  nn.ReLU(inplace=True),
                  nn.MaxPool2d(2),
                  nn.Flatten(),
                  nn.Linear(64 * 8 * 8, 512),
                  nn.ReLU(inplace=True),
                  nn.Linear(512, target_size)]
        self.blocks = nn.Sequential(*blocks)

    def f(self, x):
        x = self.blocks(x)
        return x

    def forward(self, input):
        output = {}
        output['target'] = self.f(input['data'])
        if 'loss_mode' in input:
            if input['loss_mode'] == 'sup':
                output['loss'] = loss_fn(output['target'], input['target'])
            elif input['loss_mode'] == 'fix':
                aug_output = self.f(input['aug'])
                output['loss'] = loss_fn(aug_output, input['target'].detach())
            elif input['loss_mode'] == 'fix-mix':
                aug_output = self.f(input['aug'])
                output['loss'] = loss_fn(aug_output, input['target'].detach())
                mix_output = self.f(input['mix_data'])
                output['loss'] += input['lam'] * loss_fn(mix_output, input['mix_target'][:, 0].detach()) + (
                        1 - input['lam']) * loss_fn(mix_output, input['mix_target'][:, 1].detach())
        else:
            if not torch.any(input['target'] == -1):
                output['loss'] = loss_fn(output['target'], input['target'])
        return output


def conv(track=False):
    data_shape = cfg['data_shape']
    hidden_size = cfg['conv']['hidden_size']
    target_size = cfg['target_size']
    model = Conv(data_shape, hidden_size, target_size)
    model.apply(init_param)
    model.apply(lambda m: make_batchnorm(m, momentum=None, track_running_stats=False))
    return model