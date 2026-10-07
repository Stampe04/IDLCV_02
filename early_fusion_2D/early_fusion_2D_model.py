import torch
import torch.nn as nn


class EarlyFusion2D(nn.Module):
    def __init__(self, num_classes=10, num_frames=10):
        super().__init__()

        # Early fusion:
        self.conv1 = nn.Conv2d(
            in_channels=3 * num_frames,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(128)

        # Avoid hard-coding H and W
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Expected input:
        # x.shape = [B, C, T, H, W]

        B, C, T, H, W = x.shape

        # Combine time dimension with channel dimension:
        # [B, 3, 10, H, W] -> [B, 30, H, W]
        x = x.reshape(B, C * T, H, W)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.adaptive_pool(x)

        # [B, 128, 1, 1] -> [B, 128]
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x