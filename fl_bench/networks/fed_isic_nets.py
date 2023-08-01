import torch
import torch.nn as nn
from monai.networks.nets.efficientnet import EfficientNetBN
import re


def efficientnet(norm="batch"):

    assert norm in ["batch", "layer", "group", "instance"]
    net = EfficientNetBN(
        "efficientnet-b0",
        pretrained=False,
        num_classes=8,
        norm=norm
    )
    pretrain_state_dict = torch.load('/path/to/.cache/efficientnet-b0-355c32eb.pth')
    model_state_dict = net.state_dict()
    pattern = re.compile(r"(.+)\.\d+(\.\d+\..+)")
    for key, value in model_state_dict.items():
        pretrain_key = re.sub(pattern, r"\1\2", key)
        if pretrain_key in pretrain_state_dict and value.shape == pretrain_state_dict[pretrain_key].shape:
            model_state_dict[key] = pretrain_state_dict[pretrain_key]

    net.load_state_dict(model_state_dict)

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
