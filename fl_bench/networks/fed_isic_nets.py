import torch
import torch.nn as nn
import os
import timm
import torchvision.models as tm
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.layers import BatchNormAct2d, LayerNormAct2d, LayerNorm2d

from fl_bench.fed_isic.src.utils import seed_everything
from functools import partial


model_dir = "/path/to/model_weights"
ssl_model_dir = "/path/to/FL_bench/fl_bench/skin_ssl/models"
torch.hub.set_dir(model_dir)

NUM_CLASSES = 8


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

def resnet_18_scratch(seed):
    seed_everything(seed)
    bb = tm.resnet18()
    num_ftrs = bb.fc.in_features
    bb.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb


def resnet_18_supervised(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    bb = tm.resnet18(weights="IMAGENET1K_V1")
    os.environ['https_proxy'] = ''
    num_ftrs = bb.fc.in_features
    bb.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb

def resnet_18_ssl(seed):
    seed_everything(seed)
    model = tm.resnet18()
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(
        torch.load(
            os.path.join(ssl_model_dir,
                         "resnet_18.pth"
                         )
        )
    )

    return model

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

resnet_50_scratch = resnet_50

def resnet_50_pretrained_pt(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    bb = tm.resnet50(weights="IMAGENET1K_V2")
    os.environ['https_proxy'] = ''
    num_ftrs = bb.fc.in_features
    bb.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb

resnet_50_supervised = resnet_50_pretrained_pt

def resnet_50_pretrained_ssl(seed):
    seed_everything(seed)
    model = tm.resnet50()
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(
        torch.load(
            os.path.join(ssl_model_dir,
                         "resnet_50.pth"
                         )
        )
    )

    return model

resnet_50_ssl = resnet_50_pretrained_ssl


def resnet_50_pretrained_timm(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    bb = timm.create_model(
        'resnet50.a1_in1k',
        pretrained=True,
        num_classes=NUM_CLASSES)
    os.environ['https_proxy'] = ''
    return bb


def resnet_50_pretrained_dino(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50',
                        pretrained=True)
    os.environ['https_proxy'] = ''
    model.fc = nn.Linear(2048, NUM_CLASSES)
    return model

def resnet_50_pretrained_mocov3(seed):
    seed_everything(seed)
    model = tm.resnet50()
    checkpoint = torch.load(f'{model_dir}/r-50-1000ep.pth.tar', map_location="cpu")
    state_dict = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % 'fc'):
            # remove prefix
            state_dict[k[len("module.base_encoder."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    model.load_state_dict(state_dict, strict=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model

# DENSENETS

def densenet_121(seed):
    seed_everything(seed)
    bb = tm.densenet121()
    num_ftrs = bb.classifier.in_features
    bb.classifier = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb

densenet_121_scratch = densenet_121

def densenet_121_pretrained_pt(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    bb = tm.densenet121(weights="IMAGENET1K_V1")
    os.environ['https_proxy'] = ''
    num_ftrs = bb.classifier.in_features
    bb.classifier = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb

densenet_121_supervised = densenet_121_pretrained_pt

def densenet_121_pretrained_ssl(seed):
    seed_everything(seed)
    model = tm.densenet121()
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    model.load_state_dict(
        torch.load(
            os.path.join(ssl_model_dir,
                         "densenet_121.pth"
                         )
        )
    )

    return model

densenet_121_ssl = densenet_121_pretrained_ssl

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

convnext_small_scratch = convnext_small

def convnext_small_pretrained_pt(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    bb = tm.convnext_small(weights="IMAGENET1K_V1")
    os.environ['https_proxy'] = ''
    num_ftrs = bb.classifier[2].in_features
    bb.classifier[2] = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb

convnext_small_supervised = convnext_small_pretrained_pt

def convnext_small_pretrained_timm(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    bb = timm.create_model(
        'convnext_small.fb_in1k',
        pretrained=True,
        num_classes=NUM_CLASSES)
    os.environ['https_proxy'] = ''
    return bb


def convnext_small_pretrained_ssl(seed):
    seed_everything(seed)
    model = tm.convnext_small()
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, NUM_CLASSES)
    model.load_state_dict(
        torch.load(
            os.path.join(ssl_model_dir,
                         "convnext_small.pth"
                         )
        )
    )

    return model

convnext_small_ssl = convnext_small_pretrained_ssl

# EFFICIENTNETv2 (better than ViT)


def efficientnet_v2_s(seed):
    seed_everything(seed)
    bb = tm.efficientnet_v2_s()
    num_ftrs = bb.classifier[1].in_features
    bb.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb

efficientnet_v2_s_scratch = efficientnet_v2_s

def efficientnet_v2_s_pretrained_pt(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    bb = tm.efficientnet_v2_s(weights="IMAGENET1K_V1")
    os.environ['https_proxy'] = ''
    num_ftrs = bb.classifier[1].in_features
    bb.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb

efficientnet_v2_s_supervised = efficientnet_v2_s_pretrained_pt

def efficientnet_v2_s_pretrained_ssl(seed):
    seed_everything(seed)
    model = tm.efficientnet_v2_s()
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        NUM_CLASSES
    )
    model.load_state_dict(
        torch.load(
            os.path.join(ssl_model_dir,
                         "efficientnet_v2_s_214.pth"
                         )
        )
    )

    return model

efficientnet_v2_s_ssl = efficientnet_v2_s_pretrained_ssl

# WIDE RESNET


def wide_resnet_50_2(seed):
    seed_everything(seed)
    bb = tm.wide_resnet50_2()
    num_ftrs = bb.fc.in_features
    bb.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb

wide_resnet_50_2_scratch = wide_resnet_50_2

def wide_resnet_50_2_pretrained_pt(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    bb = tm.wide_resnet50_2(weights="IMAGENET1K_V2")
    os.environ['https_proxy'] = ''
    num_ftrs = bb.fc.in_features
    bb.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb

wide_resnet_50_2_supervised = wide_resnet_50_2_pretrained_pt

def wide_resnet_50_2_pretrained_ssl(seed):
    seed_everything(seed)
    model = tm.wide_resnet50_2()
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(
        torch.load(
            os.path.join(ssl_model_dir,
                         "wide_resnet_50_2.pth"
                         )
        )
    )

    return model

wide_resnet_50_2_ssl = wide_resnet_50_2_pretrained_ssl

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

nf_resnet50_scratch = nf_resnet50


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

nf_resnet50_supervised = nf_resnet50_pretrained

def nf_resnet50_pretrained_ssl(seed):
    seed_everything(seed)
    model = timm.create_model(
        'nf_resnet50.ra2_in1k',
        pretrained=False,
        num_classes=NUM_CLASSES
    )
    model.load_state_dict(
        torch.load(
            os.path.join(ssl_model_dir,
                         "nf_resnet50.pth"
                         )
        )
    )

    return model

nf_resnet50_ssl = nf_resnet50_pretrained_ssl

# SWIN


def swin_t(seed):
    seed_everything(seed)
    bb = tm.swin_t()
    num_ftrs = bb.head.in_features
    bb.head = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb

swin_t_scratch = swin_t

def swin_t_pretrained_pt(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    bb = tm.swin_t(weights="IMAGENET1K_V1")
    os.environ['https_proxy'] = ''
    num_ftrs = bb.head.in_features
    bb.head = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb

swin_t_supervised = swin_t_pretrained_pt

def swin_t_pretrained_timm(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    bb = timm.create_model(
        'swin_tiny_patch4_window7_224.ms_in1k',
        pretrained=True,
        num_classes=NUM_CLASSES
    )
    os.environ['https_proxy'] = ''
    return bb


def swin_t_pretrained_ssl(seed):
    seed_everything(seed)
    model = tm.swin_t()
    model.head = nn.Linear(model.head.in_features, NUM_CLASSES)
    model.load_state_dict(
        torch.load(
            os.path.join(ssl_model_dir,
                         "swin_t.pth"
                         )
        )
    )

    return model

swin_t_ssl = swin_t_pretrained_ssl

# VIT


def vit_b_16(seed):
    seed_everything(seed)
    bb = tm.vit_b_16()
    num_ftrs = bb.heads.head.in_features
    bb.heads.head = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb

vit_b_16_scratch = vit_b_16

def vit_b_16_pretrained_pt(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    bb = tm.vit_b_16(weights="IMAGENET1K_V1")
    os.environ['https_proxy'] = ''
    num_ftrs = bb.heads.head.in_features
    bb.heads.head = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb

vit_b_16_supervised = vit_b_16_pretrained_pt

def vit_b_16_pretrained_ssl(seed):
    seed_everything(seed)
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16',
                           pretrained=False)
    model.head = nn.Linear(model.embed_dim, NUM_CLASSES)
    model.load_state_dict(torch.load(os.path.join(ssl_model_dir,
                                                  "vit_b_16_305.pth")))
    return model


vit_b_16_ssl = vit_b_16_pretrained_ssl

def vit_b_16_pretrained_dino(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16',
                           pretrained=True)
    model.head = nn.Linear(model.embed_dim, NUM_CLASSES)
    os.environ['https_proxy'] = ''
    return model


# MoCoV3

class VisionTransformerMoCo(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Use fixed 2D sin-cos position embedding
        self.build_2d_sincos_position_embedding()

    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False

def vit_b_16_pretrained_mocov3(seed):
    seed_everything(seed)
    model = VisionTransformerMoCo(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    model.default_cfg = _cfg()
    checkpoint = torch.load(f'{model_dir}/vit-b-300ep.pth.tar', map_location="cpu")
    state_dict = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % "head"):
            # remove prefix
            state_dict[k[len("module.base_encoder."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    model.load_state_dict(state_dict, strict=False)

    model.head = nn.Linear(model.embed_dim, NUM_CLASSES)
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


def nfnet_f0(seed): # 71.5M parameters 7.2 GMACs
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    model = timm.create_model(
        'dm_nfnet_f0.dm_in1k',
        pretrained=True,
        num_classes=NUM_CLASSES
        )
    os.environ['https_proxy'] = ''
    return model

nfnet_f0_supervised = nfnet_f0

def nfnet_l0(seed): # 35M parameters 4.4.GMAcs
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    model = timm.create_model(
        'nfnet_l0.ra2_in1k',
        pretrained=True,
        num_classes=NUM_CLASSES
        )
    os.environ['https_proxy'] = ''
    return model

nfnet_l0_supervised = nfnet_l0

def eca_nfnet_l0(seed): # 24M parameters 4.4 GMACs
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    model = timm.create_model(
        'eca_nfnet_l0.ra2_in1k',
        pretrained=True,
        num_classes=NUM_CLASSES
        )
    os.environ['https_proxy'] = ''
    return model

eca_nfnet_l0_supervised = eca_nfnet_l0


def eca_nfnet_l1(seed): # 24M parameters 4.4 GMACs
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    model = timm.create_model(
        'eca_nfnet_l1.ra2_in1k',
        pretrained=True,
        num_classes=NUM_CLASSES
        )
    os.environ['https_proxy'] = ''
    return model

eca_nfnet_l1_supervised = eca_nfnet_l1

def eca_nfnet_l2(seed): # 24M parameters 4.4 GMACs
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    model = timm.create_model(
        'eca_nfnet_l2.ra3_in1k',
        pretrained=True,
        num_classes=NUM_CLASSES
        )
    os.environ['https_proxy'] = ''
    return model

eca_nfnet_l2_supervised = eca_nfnet_l2


def silu_to_gelu(model):
    for name, child in model.named_children():

        if isinstance(child, nn.SiLU):
            setattr(model, name, nn.GELU())

        else:
            silu_to_gelu(child)




def gelu_eca_nfnet_l0(seed): # 24M parameters 4.4 GMACs
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    model = timm.create_model(
        'eca_nfnet_l0.ra2_in1k',
        pretrained=True,
        num_classes=NUM_CLASSES
        )
    os.environ['https_proxy'] = ''

    silu_to_gelu(model)

    return model

gelu_eca_nfnet_l0_supervised = gelu_eca_nfnet_l0

def gelu_eca_nfnet_l1(seed): # 24M parameters 4.4 GMACs
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    model = timm.create_model(
        'eca_nfnet_l1.ra2_in1k',
        pretrained=True,
        num_classes=NUM_CLASSES
        )
    os.environ['https_proxy'] = ''

    silu_to_gelu(model)

    return model

gelu_eca_nfnet_l1_supervised = gelu_eca_nfnet_l1

def gelu_eca_nfnet_l2(seed): # 24M parameters 4.4 GMACs
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    model = timm.create_model(
        'eca_nfnet_l2.ra3_in1k',
        pretrained=True,
        num_classes=NUM_CLASSES
        )
    os.environ['https_proxy'] = ''

    silu_to_gelu(model)

    return model

gelu_eca_nfnet_l2_supervised = gelu_eca_nfnet_l2

def gelu_nfnet_l0(seed): # 24M parameters 4.4 GMACs
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    model = timm.create_model(
        'nfnet_l0.ra2_in1k',
        pretrained=True,
        num_classes=NUM_CLASSES
        )
    os.environ['https_proxy'] = ''

    silu_to_gelu(model)

    return model

gelu_nfnet_l0_supervised = gelu_nfnet_l0


# Credit to Sara Pieri for the following


def coatnet_ln(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    model = timm.create_model(
        'coatnet_0_rw_224.sw_in1k',
        pretrained=True,
        num_classes=NUM_CLASSES
        )
    os.environ['https_proxy'] = ''
    coatnet_to_layer_norm(model)

    return model

coatnet_ln_supervised = coatnet_ln

def coatnet_to_layer_norm(module: torch.nn.Module):

    for name, child in module.named_children():

        if isinstance(child, BatchNormAct2d):
            num_channels = child.num_features
            setattr(module, name, LayerNormAct2d(num_channels,eps=1e-5,act_layer=nn.SiLU))
            
        elif isinstance(child, nn.BatchNorm2d):
            child: nn.BatchNorm2d = child
            num_channels = child.num_features
            setattr(module, name, LayerNorm2d(num_channels))

        else:
            coatnet_to_layer_norm(child)


# CoAtNeXt - A timm specific arch that uses ConvNeXt blocks in place of MBConv blocks in CoAtNet. 
# All normalization layers are LayerNorm (no BatchNorm).


def coatnetx_nano(seed): # 14.7M Parameters 2.5 GMACs
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    model = timm.create_model(
        'coatnext_nano_rw_224.sw_in1k',
        pretrained=True,
        num_classes=11
        )
    os.environ['https_proxy'] = ''
    return model

coatnetx_nano_supervised = coatnetx_nano