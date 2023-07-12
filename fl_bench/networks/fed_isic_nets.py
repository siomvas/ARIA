import torch
import torch.nn as nn
from monai.networks.nets.efficientnet import EfficientNetBN


def efficientnet(norm="batch"):

    assert norm in ["batch", "layer", "group", "instance"]

    net = EfficientNetBN(
        "efficientnet-b0",
        pretrained=True,
        num_classes=8,
        norm=norm
    )
    return net


def efficientnet_scratch(norm="batch"):

    assert norm in ["batch", "layer", "group", "instance"]

    net = EfficientNetBN(
        "efficientnet-b0",
        pretrained=False,
        num_classes=8,
        norm=norm
    )
    return net
