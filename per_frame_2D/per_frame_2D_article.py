import torch
import torch.nn as nn

class PerFrame2D(nn.Module):
    def __init__(self, num_classes=10, num_frames=10):
        super().__init__()
        self.num_frames = num_frames

        # architecture from article  
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=96,
                kernel_size=11,
                stride=3
            ),
            nn.ReLU(),
            nn.LocalResponseNorm(
                size=5,
                alpha=1e-4,
                beta=0.5,
                k=2.0
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=96,
                out_channels=256,
                kernel_size=5,
                stride=1
            ),
            nn.ReLU(),
            nn.LocalResponseNorm(
                size=5,
                alpha=1e-4,
                beta=0.5,
                k=2.0
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=256,
                out_channels=384,
                kernel_size=3,
                stride=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=384,
                out_channels=384,
                kernel_size=3,
                stride=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=384,
                out_channels=256,
                kernel_size=3,
                stride=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),

            nn.Linear(4096, 4096),
            nn.ReLU(),

            nn.Linear(4096, num_classes),
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