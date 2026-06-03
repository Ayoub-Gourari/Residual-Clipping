"""Model factory for centralized CIFAR experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16


def batch_norm(num_features: int) -> nn.Module:
    return nn.BatchNorm2d(
        num_features=num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    )


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Module:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlockV1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = batch_norm(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = batch_norm(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x) if self.downsample is not None else x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.relu(out)
        return out


class ResNetCIFAR(nn.Module):
    def __init__(self, resnet_size: int, num_classes: int = 10):
        super().__init__()
        if resnet_size % 6 != 2:
            raise ValueError(f"resnet_size must be 6n+2, got {resnet_size}.")
        num_blocks = (resnet_size - 2) // 6
        self.prep = nn.Sequential(
            conv3x3(3, 16, stride=1),
            batch_norm(16),
            nn.ReLU(inplace=False),
        )
        self.conv_1 = self._make_layer(BasicBlockV1, 16, 16, num_blocks, init_stride=1)
        self.conv_2 = self._make_layer(BasicBlockV1, 16, 32, num_blocks, init_stride=2)
        self.conv_3 = self._make_layer(BasicBlockV1, 32, 64, num_blocks, init_stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=8)
        self.classifier = nn.Linear(64, num_classes, bias=True)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(
        self,
        block: type[BasicBlockV1],
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        init_stride: int = 1,
    ) -> nn.Sequential:
        if init_stride == 1:
            downsample = None
        else:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=init_stride, bias=False),
                batch_norm(out_channels),
            )
        layers = [block(in_channels, out_channels, stride=init_stride, downsample=downsample)]
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prep(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class PreActBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + shortcut


class ResNet18CIFAR10(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
        )
        self.layers = nn.Sequential(
            self._make_layer(64, 64, 2, stride=1),
            self._make_layer(64, 128, 2, stride=2),
            self._make_layer(128, 256, 2, stride=2),
            self._make_layer(256, 256, 2, stride=2),
        )
        self.classifier = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        current_channels = in_channels
        for current_stride in strides:
            layers.append(PreActBlock(current_channels, out_channels, stride=current_stride))
            current_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prep(x)
        x = self.layers(x)
        x_avg = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x_max = F.adaptive_max_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = torch.cat([x_avg, x_max], dim=-1)
        return self.classifier(x)


def make_vgg16_cifar(num_classes: int = 10) -> nn.Module:
    model = vgg16(weights=None)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.classifier = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(inplace=False),
        nn.Dropout(p=0.5),
        nn.Linear(512, 512),
        nn.ReLU(inplace=False),
        nn.Dropout(p=0.5),
        nn.Linear(512, num_classes),
    )
    return model


def make_vgg16_cifar10(num_classes: int = 10) -> nn.Module:
    return make_vgg16_cifar(num_classes=num_classes)


def get_model(model_name: str, num_classes: int = 10) -> nn.Module:
    if model_name == "resnet20":
        return ResNetCIFAR(20, num_classes=num_classes)
    if model_name == "resnet18":
        return ResNet18CIFAR10(num_classes=num_classes)
    if model_name == "vgg16":
        return make_vgg16_cifar(num_classes=num_classes)
    raise ValueError(f"Unsupported model: {model_name}")
