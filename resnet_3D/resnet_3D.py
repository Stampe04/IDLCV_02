import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, kernel_size=3, stride=1, padding=1):
    """3x3x3 convolution with padding."""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1x1 convolution."""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        # Main branch
        self.conv1 = conv3x3(
            in_planes,
            planes,
            stride=stride
        )
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(
            planes,
            planes
        )
        self.bn2 = nn.BatchNorm3d(planes)

        # Skip / identity branch
        self.downsample = None

        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                conv1x1(
                    in_planes,
                    planes * self.expansion,
                    stride=stride
                ),
                nn.BatchNorm3d(
                    planes * self.expansion
                )
            )

    def forward(self, x):
        identity = x

        # Main branch
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Adjust identity if dimensions changed
        if self.downsample is not None:
            identity = self.downsample(x)

        # Residual connection
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        # 1x1x1: reduce channels
        self.conv1 = conv1x1(
            in_planes,
            planes
        )
        self.bn1 = nn.BatchNorm3d(planes)

        # 3x3x3
        self.conv2 = conv3x3(
            planes,
            planes,
            stride=stride
        )
        self.bn2 = nn.BatchNorm3d(planes)

        # 1x1x1: expand channels
        self.conv3 = conv1x1(
            planes,
            planes * self.expansion
        )
        self.bn3 = nn.BatchNorm3d(
            planes * self.expansion
        )

        self.relu = nn.ReLU(inplace=True)

        # Skip / identity branch
        self.downsample = None

        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                conv1x1(
                    in_planes,
                    planes * self.expansion,
                    stride=stride
                ),
                nn.BatchNorm3d(
                    planes * self.expansion
                )
            )

    def forward(self, x):
        identity = x

        # Main branch
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Adjust identity if dimensions changed
        if self.downsample is not None:
            identity = self.downsample(x)

        # Residual connection
        out += identity
        out = self.relu(out)

        return out


class ResNet_3D(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=10,
        in_channels=3
    ):
        super().__init__()

        self.in_planes = 64

        # Initial convolution
        self.conv1 = nn.Conv3d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(
            kernel_size=3,
            stride=2,
            padding=1
        )

        # ResNet stages
        self.layer1 = self._make_layer(
            block,
            64,
            layers[0],
            stride=1
        )

        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2
        )

        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2
        )

        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2
        )

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc = nn.Linear(
            512 * block.expansion,
            num_classes
        )

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1
    ):
        layers = []

        # First block may change:
        # - spatial resolution
        # - number of channels
        layers.append(
            block(
                self.in_planes,
                planes,
                stride=stride
            )
        )

        # Update channel count for subsequent blocks
        self.in_planes = planes * block.expansion

        # Remaining blocks keep same dimensions
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride=1
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):

        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Pool + classifier
        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x