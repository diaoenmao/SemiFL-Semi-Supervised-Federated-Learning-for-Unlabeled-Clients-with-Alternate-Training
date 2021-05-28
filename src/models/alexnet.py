import torch
import numpy as np
from config import cfg
from .utils import init_param, make_batchnorm, loss_fn


def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))


class AlexNet(torch.nn.Module):
    def __init__(self, data_shape, target_size):
        super().__init__()
        ncha, size, _ = data_shape
        self.target_size = target_size
        self.conv1 = torch.nn.Conv2d(ncha, 64, kernel_size=size // 8)
        s = compute_conv_output_size(size, size // 8)
        s = s // 2
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=size // 10)
        s = compute_conv_output_size(s, size // 10)
        s = s // 2
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=2)
        s = compute_conv_output_size(s, 2)
        s = s // 2
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(256 * s * s, 2048)
        self.fc2 = torch.nn.Linear(2048, 2048)
        self.last = torch.nn.Linear(2048, target_size)

    def f(self, x):
        h = self.maxpool(self.drop1(self.relu(self.conv1(x))))
        h = self.maxpool(self.drop1(self.relu(self.conv2(h))))
        h = self.maxpool(self.drop2(self.relu(self.conv3(h))))
        h = h.view(x.size(0), -1)
        h = self.drop2(self.relu(self.fc1(h)))
        h = self.drop2(self.relu(self.fc2(h)))
        y = self.last(h)
        return y

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


def alexnet():
    data_shape = cfg['data_shape']
    target_size = cfg['target_size']
    model = AlexNet(data_shape, target_size)
    model.apply(init_param)
    model.apply(lambda m: make_batchnorm(m, momentum=None, track_running_stats=False))
    return model
