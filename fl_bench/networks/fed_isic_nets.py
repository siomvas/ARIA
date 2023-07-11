import torch
import torch.nn as nn
from monai.networks.nets.efficientnet import EfficientNetBN


def efficientnet(pretrained=False, norm="batch"):

    assert norm in ["batch", "layer", "group", "instance"]

    net = EfficientNetBN(
        "efficientnet-b0",
        pretrained=pretrained,
        num_classes=8,
        norm=norm
    )
    return net
