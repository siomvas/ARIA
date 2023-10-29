import torch
import torch.nn as nn
from monai.networks.nets.efficientnet import EfficientNetBN
import re
import os
import timm
import torchvision.models as tm
from fl_bench.fed_isic.src.utils import seed_everything


model_dir = "/path/to/model_weights"
skin_ssl_model_dir = "/path/to/FL_bench/fl_bench/skin_ssl/models"
torch.hub.set_dir(model_dir)

NUM_CLASSES = 8


def efficientnet(seed=1, norm="batch"):

    assert norm in ["batch", "layer", "group", "instance"]
    net = EfficientNetBN(
        "efficientnet-b0",
        pretrained=False,
        num_classes=NUM_CLASSES,
        norm=norm
    )
    pretrain_state_dict = torch.load(
        '/path/to/.cache/efficientnet-b0-355c32eb.pth'
    )
    model_state_dict = net.state_dict()
    pattern = re.compile(r"(.+)\.\d+(\.\d+\..+)")
    for key, value in model_state_dict.items():
        pretrain_key = re.sub(pattern, r"\1\2", key)
        if (pretrain_key in pretrain_state_dict and
                value.shape == pretrain_state_dict[pretrain_key].shape):
            model_state_dict[key] = pretrain_state_dict[pretrain_key]

    net.load_state_dict(model_state_dict)

    return net


def efficientnet_scratch(norm="batch"):

    assert norm in ["batch", "layer", "group", "instance"]

    net = EfficientNetBN(
        "efficientnet-b0",
        pretrained=False,
        num_classes=NUM_CLASSES,
        norm=norm
    )
    return net

# VGG


def vgg_11(seed):
    seed_everything(seed)
    bb = tm.vgg11()
    num_ftrs = bb.classifier[6].in_features
    bb.classifier[6] = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb


def vgg_11_pretrained_pt(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    bb = tm.vgg11_bn(weights="IMAGENET1K_V1")
    os.environ['https_proxy'] = ''
    num_ftrs = bb.classifier[6].in_features
    bb.classifier[6] = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb

# RESNET


def resnet_18(seed):
    seed_everything(seed)
    bb = tm.resnet18()
    num_ftrs = bb.fc.in_features
    bb.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb


def resnet_18_pretrained_pt(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    bb = tm.resnet18(weights="IMAGENET1K_V1")
    os.environ['https_proxy'] = ''
    num_ftrs = bb.fc.in_features
    bb.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb


def resnet_18_pretrained_timm(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    bb = timm.create_model(
        'resnet18.a1_in1k',
        pretrained=True,
        num_classes=NUM_CLASSES)
    os.environ['https_proxy'] = ''
    return bb


def resnet_50(seed):
    seed_everything(seed)
    bb = tm.resnet50()
    num_ftrs = bb.fc.in_features
    bb.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb


def resnet_50_pretrained_pt(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    bb = tm.resnet50(weights="IMAGENET1K_V2")
    os.environ['https_proxy'] = ''
    num_ftrs = bb.fc.in_features
    bb.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb


def resnet_50_pretrained_timm(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    bb = timm.create_model(
        'resnet50.a1_in1k',
        pretrained=True,
        num_classes=NUM_CLASSES)
    os.environ['https_proxy'] = ''
    return bb


# DENSENETS

def densenet_121(seed):
    seed_everything(seed)
    bb = tm.densenet121()
    num_ftrs = bb.classifier.in_features
    bb.classifier = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb


def densenet_121_pretrained_pt(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    bb = tm.densenet121(weights="IMAGENET1K_V1")
    os.environ['https_proxy'] = ''
    num_ftrs = bb.classifier.in_features
    bb.classifier = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb


def densenet_169(seed):
    seed_everything(seed)
    bb = tm.densenet169()
    num_ftrs = bb.classifier.in_features
    bb.classifier = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb


def densenet_169_pretrained_pt(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    bb = tm.densenet169(weights="IMAGENET1K_V1")
    os.environ['https_proxy'] = ''
    num_ftrs = bb.classifier.in_features
    bb.classifier = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb


# CONVNEXT

def convnext_small(seed):
    seed_everything(seed)
    bb = tm.convnext_small()
    num_ftrs = bb.classifier[2].in_features
    bb.classifier[2] = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb


def convnext_small_pretrained_pt(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    bb = tm.convnext_small(weights="IMAGENET1K_V1")
    os.environ['https_proxy'] = ''
    num_ftrs = bb.classifier[2].in_features
    bb.classifier[2] = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb

# EFFICIENTNETv2 (better than ViT)


def efficientnet_v2_s(seed):
    seed_everything(seed)
    bb = tm.efficientnet_v2_s()
    num_ftrs = bb.classifier[1].in_features
    bb.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb


def efficientnet_v2_s_pretrained_pt(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    bb = tm.efficientnet_v2_s(weights="IMAGENET1K_V1")
    os.environ['https_proxy'] = ''
    num_ftrs = bb.classifier[1].in_features
    bb.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb


def efficientnet_v2_s_ssl(seed):
    seed_everything(seed)
    model = tm.efficientnet_v2_s()
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        NUM_CLASSES
    )
    model.load_state_dict(
        torch.load(
            os.path.join(skin_ssl_model_dir,
                         "efficientnet_v2_s_214.pth"
                         )
        )
    )

    return model

# WIDE RESNET


def wide_resnet_50_2(seed):
    seed_everything(seed)
    bb = tm.wide_resnet50_2()
    num_ftrs = bb.fc.in_features
    bb.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb


def wide_resnet_50_2_pretrained_pt(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    bb = tm.wide_resnet50_2(weights="IMAGENET1K_V2")
    os.environ['https_proxy'] = ''
    num_ftrs = bb.fc.in_features
    bb.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb


# RESNEXT

def resnext50_32x4d(seed):
    seed_everything(seed)
    bb = tm.resnext50_32x4d()
    num_ftrs = bb.fc.in_features
    bb.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb


def resnext_50_32x4d_pretrained_pt(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    bb = tm.resnext50_32x4d(weights="IMAGENET1K_V2")
    os.environ['https_proxy'] = ''
    num_ftrs = bb.fc.in_features
    bb.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb


# NORM FREE RESNET

def nf_resnet50(seed):
    seed_everything(seed)
    model = timm.create_model(
        'nf_resnet50.ra2_in1k',
        pretrained=False,
        num_classes=NUM_CLASSES
    )
    return model


def nf_resnet50_pretrained(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    model = timm.create_model(
        'nf_resnet50.ra2_in1k',
        pretrained=True,
        num_classes=NUM_CLASSES
    )
    os.environ['https_proxy'] = ''
    return model


# SWIN


def swin_t(seed):
    seed_everything(seed)
    bb = tm.swin_t()
    num_ftrs = bb.head.in_features
    bb.head = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb


def swin_t_pretrained_pt(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    bb = tm.swin_t(weights="IMAGENET1K_V1")
    os.environ['https_proxy'] = ''
    num_ftrs = bb.head.in_features
    bb.head = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb


# VIT


def vit_b_16(seed):
    seed_everything(seed)
    bb = tm.vit_b_16()
    num_ftrs = bb.heads.head.in_features
    bb.heads.head = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb


def vit_b_16_pretrained_pt(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    bb = tm.vit_b_16(weights="IMAGENET1K_V1")
    os.environ['https_proxy'] = ''
    num_ftrs = bb.heads.head.in_features
    bb.heads.head = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb


def vit_b_16_pretrained_mae_ssl(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    model = timm.create_model(
        'vit_base_patch16_224.mae',
        pretrained=True,
        num_classes=NUM_CLASSES
    )
    os.environ['https_proxy'] = ''
    return model


def vit_b_16_pretrained_ssl(seed):
    seed_everything(seed)
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16',
                           pretrained=False)
    model.head = nn.Linear(model.embed_dim, NUM_CLASSES)
    model.load_state_dict(torch.load(os.path.join(skin_ssl_model_dir,
                                                  "vit_b_16_305.pth")))
    return model


def vit_b_14_pretrained_dino2(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    model = timm.create_model(
        'vit_base_patch14_dinov2.lvd142m',
        pretrained=True,
        num_classes=NUM_CLASSES
    )
    os.environ['https_proxy'] = ''
    return model

# ConvMixer


def convmixer(seed):
    seed_everything(seed)
    model = timm.create_model(
        'convmixer_768_32.in1k',
        pretrained=False,
        num_classes=NUM_CLASSES
    )
    return model


def convmixer_pretrained(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    model = timm.create_model(
        'convmixer_768_32.in1k',
        pretrained=True,
        num_classes=NUM_CLASSES
    )
    os.environ['https_proxy'] = ''
    return model


# get model specific transforms (normalization, resize)
# data_config = timm.data.resolve_model_data_config(model)
# transforms = timm.data.create_transform(**data_config, is_training=False)
