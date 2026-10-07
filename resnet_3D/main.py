import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
#import dataloader
from dataloader import get_video_loaders
if __package__:
    from .resnet_3D import ResNet_3D, BasicBlock
else:
    from resnet_3D import ResNet_3D, BasicBlock
from train import train_model

# train_loader, val_loader, test_loader = get_video_loaders()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Equivalent structure to ResNet-18
model = ResNet_3D(
    BasicBlock,
    [2, 2, 2, 2],
    num_classes=10,
    in_channels=3
).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_loader, val_loader, test_loader = get_video_loaders()

train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=20,
    device=device
)