import torch
import torch.nn as nn


def inflate_resnet18_weights(model_3d, model_2d):
    """
    Initialize a 3D ResNet-18 from pretrained 2D ResNet-18 weights.
    """

    state_2d = model_2d.state_dict()
    state_3d = model_3d.state_dict()

    transferred = []
    skipped = []

    for name, weights_3d in state_3d.items():

        if name not in state_2d:
            skipped.append(name)
            continue

        weights_2d = state_2d[name]

        # Convert Conv2D kernels into Conv3D kernels
        if weights_2d.ndim == 4 and weights_3d.ndim == 5:

            temporal_size = weights_3d.shape[2]

            inflated = (
                weights_2d.unsqueeze(2)
                .repeat(1, 1, temporal_size, 1, 1)
                / temporal_size
            )

            if inflated.shape == weights_3d.shape:
                state_3d[name] = inflated
                transferred.append(name)
            else:
                skipped.append(name)

        # Transfer BatchNorm and other matching parameters
        elif weights_2d.shape == weights_3d.shape:

            # Initialize a new classifier for video classes
            if name.startswith("fc."):
                skipped.append(name)
                continue

            state_3d[name] = weights_2d.clone()
            transferred.append(name)

        else:
            skipped.append(name)

    model_3d.load_state_dict(state_3d)

    print(f"Transferred {len(transferred)} tensors")
    print(f"Skipped {len(skipped)} tensors")
    print("Skipped:", skipped)

    return model_3d


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
        in_channels=3,
        base_channels=64,
        dropout=0.0
    ):
        super().__init__()

        if len(layers) != 4:
            raise ValueError("layers must contain four stage depths")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0)")

        self.in_planes = base_channels

        # Initial convolution
        self.conv1 = nn.Conv3d(
            in_channels,
            base_channels,
            kernel_size=7,
            stride=2,
            padding=2,
            bias=False
        )

        self.bn1 = nn.BatchNorm3d(base_channels)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(
            kernel_size=3,
            stride=2,
            padding=1
        )

        # ResNet stages
        self.layer1 = self._make_layer(
            block,
            base_channels,
            layers[0],
            stride=1
        )

        self.layer2 = self._make_layer(
            block,
            base_channels * 2,
            layers[1],
            stride=2
        )

        self.layer3 = self._make_layer(
            block,
            base_channels * 4,
            layers[2],
            stride=2
        )

        self.layer4 = self._make_layer(
            block,
            base_channels * 8,
            layers[3],
            stride=2
        )

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=dropout)

        self.fc = nn.Linear(
            base_channels * 8 * block.expansion,
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
        x = self.dropout(x)
        x = self.fc(x)

        return x