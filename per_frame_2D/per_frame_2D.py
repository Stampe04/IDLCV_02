import torch
import torch.nn as nn

class PerFrame2D(nn.Module):
    def __init__(self, num_classes=10, num_frames=10):
        super().__init__()
        self.num_frames = num_frames
        self.fc = nn.Linear(128, num_classes)

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, T, H, W = x.shape

        x = x.permute(0, 2, 1, 3, 4)  
        x = x.reshape(B * T, C, H, W) 

        x = self.model(x)

        x = self.fc(x)
        x = x.reshape(B, T, -1)
        x = x.mean(dim=1) # average of predictions

        x = self.softmax(x)

        return x

if __name__ == "__main__":
    m = PerFrame2D(num_classes=10, num_frames=10)
    print(m(torch.randn(2, 3, 10, 64, 64)).shape)