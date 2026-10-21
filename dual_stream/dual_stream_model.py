import torch
import torch.nn as nn


class DualStreamNetwork(nn.Module):
    def __init__(self, num_classes=10, num_frames=10):
        super().__init__()

        self.temporal_model = nn.Sequential(
            nn.Conv2d(
                in_channels=2 * (num_frames -1),
                out_channels=96,
                kernel_size=7,
                stride = 2
            ),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=96,
                out_channels=256,
                kernel_size=5,
                stride = 2
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride = 1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride = 1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride = 1
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),

            nn.LazyLinear(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(2048, num_classes))

        self.spatial_model = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=96,
                kernel_size=7,
                stride = 2
            ),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=96,
                out_channels=256,
                kernel_size=5,
                stride = 2
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride = 1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride = 1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride = 1
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),

            nn.LazyLinear(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(2048, num_classes))

    def forward(self, x_spatial, x_flow):
        # x_spatial: [B, 3, H, W]            (egyetlen frame)
        # x_flow:    [B, 2*(T-1), H, W]      (optical flow stack)
        # ha a flow [B, 2, T-1, H, W] alakú: x_flow = x_flow.flatten(1, 2)

        logits_s = self.spatial_model(x_spatial)
        logits_t = self.temporal_model(x_flow)

        return (logits_s + logits_t) / 2     