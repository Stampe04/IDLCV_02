import torch

from resnet_3D import BasicBlock, ResNet_3D


model = ResNet_3D(
    BasicBlock,
    [2, 2, 2, 2],
    num_classes=10,
    in_channels=3
)
model.eval()

videos = torch.randn(2, 3, 16, 64, 64)

with torch.no_grad():
    outputs = model(videos)

print("Input shape:", videos.shape)
print("Output shape:", outputs.shape)

assert outputs.shape == (2, 10)
print("Small forward-pass test passed")
