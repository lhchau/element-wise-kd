from .cifar100 import get_cifar100
from .cifar10 import get_cifar10
from .tiny_imagenet import get_tiny_imagenet
from .cifar100_super_class import get_cifar100_super_class
from .imagenet import *

# Data
def get_dataloader(
    data_name='cifar10',
    batch_size=256,
    num_workers=4):
    print('==> Preparing data..')

    if data_name == "cifar100":
        return get_cifar100(batch_size, num_workers)
    elif data_name == "cifar10":
        return get_cifar10(batch_size, num_workers)
    elif data_name == "tiny_imagenet":
        return get_tiny_imagenet(batch_size, num_workers)
    elif data_name == "cifar100_super_class":
        return get_cifar100_super_class(batch_size, num_workers)
    elif data_name == 'imagenet':
        return get_imagenet_dataloaders(batch_size=batch_size, val_batch_size=128, num_workers=num_workers)