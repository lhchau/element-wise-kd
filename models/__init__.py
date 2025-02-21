import torchvision
from .resnet import *
from .wideresnet import *
from .densenet import *
from .resnet_mlp import resnet18_mlp


def get_model(model_name, num_classes, widen_factor=1, dropout=None):
    if model_name == "resnet18":
        return resnet18(num_classes=num_classes, widen_factor=widen_factor, dropout=dropout)
    elif model_name == "resnet18_mlp":
        return resnet18_mlp(num_classes=num_classes, widen_factor=widen_factor, dropout=dropout)
    elif model_name == "resnet34":
        return resnet34(num_classes=num_classes, widen_factor=widen_factor)
    elif model_name == "resnet50":
        return resnet50(num_classes=num_classes, widen_factor=widen_factor)
    elif model_name == "resnet101":
        return resnet101(num_classes=num_classes, widen_factor=widen_factor)
    elif model_name == "resnet152":
        return resnet152(num_classes=num_classes, widen_factor=widen_factor)
    elif model_name == "wideresnet28_10":
        return wideresnet28_10(num_classes=num_classes)
    elif model_name == "wideresnet40_2":
        return wideresnet40_2(num_classes=num_classes)
    elif model_name == "densenet121":
        return densenet121(num_classes=num_classes)
    elif model_name == "densenet169":
        return densenet169(num_classes=num_classes)
    else:
        raise ValueError("Invalid model!!!")