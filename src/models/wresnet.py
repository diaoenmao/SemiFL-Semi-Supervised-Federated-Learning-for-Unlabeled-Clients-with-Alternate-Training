import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import init_param, make_batchnorm, loss_fn, kld_loss
from config import cfg


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equal_inout = (in_planes == out_planes)
        self.shortcut = (not self.equal_inout) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                             padding=0, bias=False) or None

    def forward(self, x):
        if not self.equal_inout:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equal_inout else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        out = torch.add(x if self.equal_inout else self.shortcut(x), out)
        return out


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer(x)
        return out


class WideResNet(nn.Module):
    def __init__(self, data_shape, num_classes, depth, widen_factor, drop_rate):
        super(WideResNet, self).__init__()
        hidden_size = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        self.conv1 = nn.Conv2d(data_shape[0], hidden_size[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, hidden_size[0], hidden_size[1], block, 1, drop_rate)
        self.block2 = NetworkBlock(n, hidden_size[1], hidden_size[2], block, 2, drop_rate)
        self.block3 = NetworkBlock(n, hidden_size[2], hidden_size[3], block, 2, drop_rate)
        self.bn1 = nn.BatchNorm2d(hidden_size[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(hidden_size[3], num_classes)
        self.hidden_size = hidden_size[3]

    def f(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.relu(self.bn1(x))
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
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


def wresnet28x2(track=False):
    data_shape = cfg['data_shape']
    target_size = cfg['target_size']
    depth = cfg['wresnet28x2']['depth']
    widen_factor = cfg['wresnet28x2']['widen_factor']
    drop_rate = cfg['wresnet28x2']['drop_rate']
    model = WideResNet(data_shape, target_size, depth, widen_factor, drop_rate)
    model.apply(init_param)
    model.apply(lambda m: make_batchnorm(m, momentum=None, track_running_stats=track))
    return model


def wresnet28x8(track=False):
    data_shape = cfg['data_shape']
    target_size = cfg['target_size']
    depth = cfg['wresnet28x8']['depth']
    widen_factor = cfg['wresnet28x8']['widen_factor']
    drop_rate = cfg['wresnet28x8']['drop_rate']
    model = WideResNet(data_shape, target_size, depth, widen_factor, drop_rate)
    model.apply(init_param)
    model.apply(lambda m: make_batchnorm(m, momentum=None, track_running_stats=track))
    return model


def wresnet37x2(track=False):
    data_shape = cfg['data_shape']
    target_size = cfg['target_size']
    depth = cfg['wresnet37x2']['depth']
    widen_factor = cfg['wresnet37x2']['widen_factor']
    drop_rate = cfg['wresnet37x2']['drop_rate']
    model = WideResNet(data_shape, target_size, depth, widen_factor, drop_rate)
    model.apply(init_param)
    model.apply(lambda m: make_batchnorm(m, momentum=None, track_running_stats=track))
    return model
