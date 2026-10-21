import torch
import torch.nn as nn


class DualStreamNetwork(nn.Module):
    def __init__(self, num_classes=10, num_frames=10):
        super().__init__()

        self.temporal_model = nn.Sequential(
            nn.Conv2d(
                in_channels=3 * (num_frames -1),
                out_channels=96,
                kernel_size=7,
                padding=3,
                stride = 2
            ),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5),

            nn.Flatten(),

            nn.LazyLinear(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1024, num_classes))

        self.spatial_model = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=96,
                kernel_size=7,
                padding=3,
                stride = 2
            ),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Dropout(0.5),

            nn.Flatten(),

            nn.LazyLinear(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(1024, num_classes))

    def forward(self, x_spatial, x_flow, w=0.33):
        logits_s = self.spatial_model(x_spatial)
        logits_t = self.temporal_model(x_flow)

        return w * logits_s + (1 - w) * logits_t