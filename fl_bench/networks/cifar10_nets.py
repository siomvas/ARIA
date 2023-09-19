import torch
import torch.nn as nn
from types import SimpleNamespace
import torch.nn.functional as F
import torch.nn.init as init
import timm
from collections import OrderedDict
from vit_pytorch import SimpleViT


class SimpleCNN_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 3 * 3, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc1(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def rep_and_out(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        features = x
        x = F.relu(x)
        x = self.fc3(x)
        return features, x


class ModerateCNN_moon(nn.Module):
    def __init__(self):
        super(ModerateCNN_moon, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),
            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            # nn.Linear(4096, 1024),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            # nn.Linear(1024, 512),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x
    
    def rep_and_out(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer[:-1](x)
        features = x
        x = self.fc_layer[-1](x)
        return features, x


# FedSAM
# c=64 k=5 ReLU -> MaxPool
# c=64 k=5 ReLU -> MaxPool
# FC n=384
# FC n=192
class ClientModel(nn.Module):
    def __init__(self, num_classes=10):
        super(ClientModel, self).__init__()
        self.num_classes = num_classes

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 5 * 5, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, self.num_classes),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.classifier(x)
        return x


# Wide Res Net from SAM repo


class BasicUnit(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super(BasicUnit, self).__init__()
        self.block = nn.Sequential(
            OrderedDict(
                [
                    ("0_normalization", nn.BatchNorm2d(channels)),
                    ("1_activation", nn.ReLU(inplace=True)),
                    (
                        "2_convolution",
                        nn.Conv2d(
                            channels, channels, (3, 3), stride=1, padding=1, bias=False
                        ),
                    ),
                    ("3_normalization", nn.BatchNorm2d(channels)),
                    ("4_activation", nn.ReLU(inplace=True)),
                    ("5_dropout", nn.Dropout(dropout, inplace=True)),
                    (
                        "6_convolution",
                        nn.Conv2d(
                            channels, channels, (3, 3), stride=1, padding=1, bias=False
                        ),
                    ),
                ]
            )
        )

    def forward(self, x):
        return x + self.block(x)


class DownsampleUnit(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int, dropout: float
    ):
        super(DownsampleUnit, self).__init__()
        self.norm_act = nn.Sequential(
            OrderedDict(
                [
                    ("0_normalization", nn.BatchNorm2d(in_channels)),
                    ("1_activation", nn.ReLU(inplace=True)),
                ]
            )
        )
        self.block = nn.Sequential(
            OrderedDict(
                [
                    (
                        "0_convolution",
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            (3, 3),
                            stride=stride,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    ("1_normalization", nn.BatchNorm2d(out_channels)),
                    ("2_activation", nn.ReLU(inplace=True)),
                    ("3_dropout", nn.Dropout(dropout, inplace=True)),
                    (
                        "4_convolution",
                        nn.Conv2d(
                            out_channels,
                            out_channels,
                            (3, 3),
                            stride=1,
                            padding=1,
                            bias=False,
                        ),
                    ),
                ]
            )
        )
        self.downsample = nn.Conv2d(
            in_channels, out_channels, (1, 1), stride=stride, padding=0, bias=False
        )

    def forward(self, x):
        x = self.norm_act(x)
        return self.block(x) + self.downsample(x)


class Block_wide(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        depth: int,
        dropout: float,
    ):
        super(Block_wide, self).__init__()
        self.block = nn.Sequential(
            DownsampleUnit(in_channels, out_channels, stride, dropout),
            *(BasicUnit(out_channels, dropout) for _ in range(depth))
        )

    def forward(self, x):
        return self.block(x)


class WideResNet(nn.Module):
    def __init__(
        self,
        depth: int,
        width_factor: int,
        dropout: float = 0.0,
        in_channels: int = 3,
        labels: int = 10,
    ):
        super(WideResNet, self).__init__()

        self.filters = [
            16,
            1 * 16 * width_factor,
            2 * 16 * width_factor,
            4 * 16 * width_factor,
        ]
        self.block_depth = (depth - 4) // (3 * 2)

        self.f = nn.Sequential(
            OrderedDict(
                [
                    (
                        "0_convolution",
                        nn.Conv2d(
                            in_channels,
                            self.filters[0],
                            (3, 3),
                            stride=1,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (
                        "1_block",
                        Block_wide(
                            self.filters[0],
                            self.filters[1],
                            1,
                            self.block_depth,
                            dropout,
                        ),
                    ),
                    (
                        "2_block",
                        Block_wide(
                            self.filters[1],
                            self.filters[2],
                            2,
                            self.block_depth,
                            dropout,
                        ),
                    ),
                    (
                        "3_block",
                        Block_wide(
                            self.filters[2],
                            self.filters[3],
                            2,
                            self.block_depth,
                            dropout,
                        ),
                    ),
                    ("4_normalization", nn.BatchNorm2d(self.filters[3])),
                    ("5_activation", nn.ReLU(inplace=True)),
                    ("6_pooling", nn.AvgPool2d(kernel_size=8)),
                    ("7_flattening", nn.Flatten()),
                    (
                        "8_classification",
                        nn.Linear(in_features=self.filters[3], out_features=labels),
                    ),
                ]
            )
        )

        self._initialize()

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight.data, mode="fan_in", nonlinearity="relu"
                )
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.zero_()
                m.bias.data.zero_()

    def forward(self, x):
        return self.f(x)

    def rep_and_out(self, x):
        x = self.f[:-1](x)
        features = x
        x = self.f[-1](x)
        return features, x


def wideresnet_16_8_moon():
    return WideResNet(depth=16, width_factor=8, dropout=0, in_channels=3, labels=10)


# MobileNet from kuangliu


class Block(nn.Module):
    """expand + depthwise + pointwise"""

    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=planes,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                OrderedDict(
                [
                    ('short_conv', nn.Conv2d(in_planes,
                                             out_planes,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0,
                                             bias=False)), 
                    ('short_bn', nn.BatchNorm2d(out_planes))
                ]))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetv2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [
        (1, 16, 1, 1),
        (6, 24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    def __init__(self, num_classes=10):
        super(MobileNetv2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(
            320, 1280, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(1280)
        self.classifier = nn.Linear(1280, num_classes)
        self.avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False, return_logits=True):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        # out = F.avg_pool2d(out, 4)
        out = self.avg_pool2d(out)

        features = out.view(out.size(0), -1)

        if not return_logits:
            return features

        out = self.classifier(features)
        if return_features:
            return features, out
        else:
            return out

    def rep_and_out(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.avg_pool2d(out)
        out = out.view(out.size(0), -1)
        features = out
        out = self.classifier(out)
        return features, out


def mobilenetv2_moon(num_classes):
    return MobileNetv2(num_classes=num_classes)


# ResNets from Bytedance


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(
                    OrderedDict([
                        ('short_conv',
                         nn.Conv2d(in_planes, self.expansion * planes,
                                   kernel_size=1, stride=stride, bias=False)),
                        ('short_norm', nn.BatchNorm2d(self.expansion * planes))
                        ])
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.classifier = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, option="A"))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        features = out
        out = self.classifier(out)
        if return_features:
            return out, features
        else:
            return out

    def rep_and_out(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        features = out
        out = self.classifier(out)
        return out, features


# This is a wider version of resnet, e.g., resnet18.
class ResNetWider(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetWider, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, option="B"))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        features = out
        out = self.classifier(out)
        if return_features:
            return out, features
        else:
            return out

    def rep_and_out(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        features = out
        out = self.classifier(out)
        return out, features


def resnet20_moon(num_classes=10):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)


def resnet32(num_classes=10):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes)


def resnet18_moon(num_classes=10):
    return ResNetWider(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


# DenseNet from pytorch lightning
act_fn_by_name = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "gelu": nn.GELU,
}


class DenseLayer(nn.Module):
    def __init__(self, c_in, bn_size, growth_rate, act_fn):
        """
        Inputs:
            c_in - Number of input channels
            bn_size - Bottleneck size (factor of growth rate) for the output of the 1x1 convolution. Typically between 2 and 4.
            growth_rate - Number of output channels of the 3x3 convolution
            act_fn - Activation class constructor (e.g. nn.ReLU)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_fn(),
            nn.Conv2d(c_in, bn_size * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(bn_size * growth_rate),
            act_fn(),
            nn.Conv2d(
                bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False
            ),
        )

    def forward(self, x):
        out = self.net(x)
        out = torch.cat([out, x], dim=1)
        return out


class DenseBlock(nn.Module):
    def __init__(self, c_in, num_layers, bn_size, growth_rate, act_fn):
        """
        Inputs:
            c_in - Number of input channels
            num_layers - Number of dense layers to apply in the block
            bn_size - Bottleneck size to use in the dense layers
            growth_rate - Growth rate to use in the dense layers
            act_fn - Activation function to use in the dense layers
        """
        super().__init__()
        layers = []
        for layer_idx in range(num_layers):
            # Input channels are original plus the feature maps from previous layers
            layer_c_in = c_in + layer_idx * growth_rate
            layers.append(
                DenseLayer(
                    c_in=layer_c_in,
                    bn_size=bn_size,
                    growth_rate=growth_rate,
                    act_fn=act_fn,
                )
            )
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return out


class TransitionLayer(nn.Module):
    def __init__(self, c_in, c_out, act_fn):
        super().__init__()
        self.transition = nn.Sequential(
            OrderedDict([
                ("trans_norm", nn.BatchNorm2d(c_in)),
                ("trans_act", act_fn()),
                ("trans_conv", nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)),
                ("trans_pool", nn.AvgPool2d(kernel_size=2, stride=2))
            ])
        )

    def forward(self, x):
        return self.transition(x)


class DenseNet(nn.Module):
    def __init__(
        self,
        num_classes=10,
        num_layers=[6, 6, 6, 6],
        bn_size=2,
        growth_rate=16,
        act_fn_name="relu",
        **kwargs
    ):
        super().__init__()
        self.hparams = SimpleNamespace(
            num_classes=num_classes,
            num_layers=num_layers,
            bn_size=bn_size,
            growth_rate=growth_rate,
            act_fn_name=act_fn_name,
            act_fn=act_fn_by_name[act_fn_name],
        )
        self._create_network()
        self._init_params()

    def _create_network(self):
        c_hidden = (
            self.hparams.growth_rate * self.hparams.bn_size
        )  # The start number of hidden channels

        # A first convolution on the original image to scale up the channel size
        self.input_net = nn.Sequential(
            # No batch norm or activation function as done inside the Dense layers
            nn.Conv2d(3, c_hidden, kernel_size=3, padding=1)
        )

        # Creating the dense blocks, eventually including transition layers
        blocks = []
        for block_idx, num_layers in enumerate(self.hparams.num_layers):
            blocks.append(
                DenseBlock(
                    c_in=c_hidden,
                    num_layers=num_layers,
                    bn_size=self.hparams.bn_size,
                    growth_rate=self.hparams.growth_rate,
                    act_fn=self.hparams.act_fn,
                )
            )
            c_hidden = (
                c_hidden + num_layers * self.hparams.growth_rate
            )  # Overall output of the dense block
            if (
                block_idx < len(self.hparams.num_layers) - 1
            ):  # Don't apply transition layer on last block
                blocks.append(
                    TransitionLayer(
                        c_in=c_hidden, c_out=c_hidden // 2, act_fn=self.hparams.act_fn
                    )
                )
                c_hidden = c_hidden // 2

        self.blocks = nn.Sequential(*blocks)

        # Mapping to classification output
        self.output_net = nn.Sequential(
            OrderedDict([
                ("out_norm", nn.BatchNorm2d(c_hidden)),
                ("out_act", self.hparams.act_fn()),
                ("out_pool", nn.AdaptiveAvgPool2d((1, 1))),
                ("out_flatten", nn.Flatten()),
                ("out_linear", nn.Linear(c_hidden, self.hparams.num_classes))
            ])
        )

    def _init_params(self):
        # Based on our discussion in Tutorial 4, we should initialize the
        # convolutions according to the activation function
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity=self.hparams.act_fn_name)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_net(x)
        x = self.blocks(x)
        x = self.output_net(x)
        return x
    
    def rep_and_out(self, x):
        x = self.input_net(x)
        x = self.blocks(x)
        x = self.output_net[:-1](x)
        features = x
        x = self.output_net[-1](x)
        return features, x


def densenet_16_moon():
    return DenseNet(num_layers=[6, 6, 6])


def swin():
    return timm.models.SwinTransformer(
        img_size=32,
        patch_size=4,
        num_classes=10,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        head_dim=None,
        window_size=4,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        proj_drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1
    )


def focalnet():
    return timm.models.FocalNet(
        num_classes=10
    )


def simple_vit_quarter():
    return SimpleViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=256,
        depth=3,
        heads=8,
        mlp_dim=512
    )


def simple_vit_half():
    return SimpleViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=512,
        depth=3,
        heads=8,
        mlp_dim=1024
    )

# Group Norm networks


class BasicBlock_group(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(BasicBlock_group, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = nn.GroupNorm(2, planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = nn.GroupNorm(2, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(
                    OrderedDict([
                        ('short_conv',
                         nn.Conv2d(in_planes, self.expansion * planes,
                                   kernel_size=1, stride=stride, bias=False)),
                        ('short_norm', nn.GroupNorm(2, self.expansion * planes))
                        ])
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_group(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_group, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(16)
        self.bn1 = nn.GroupNorm(2, 16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.classifier = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, option="A"))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        features = out
        out = self.classifier(out)
        if return_features:
            return out, features
        else:
            return out

    def rep_and_out(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        features = out
        out = self.classifier(out)
        return out, features


def resnet20_group(num_classes=10):
    return ResNet_group(BasicBlock_group, [3, 3, 3], num_classes=num_classes)
