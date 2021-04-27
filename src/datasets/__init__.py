from .mnist import MNIST
from .cifar import CIFAR10, CIFAR100
from .svhn import SVHN
from .stl import STL10
from .utils import *
from .randaugment import RandAugment

__all__ = ('MNIST', 'CIFAR10', 'CIFAR100', 'SVHN', 'STL10')
