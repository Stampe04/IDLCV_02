import torch

from dataloader import get_video_loaders
from early_fusion_2D.early_fusion_2D_model import EarlyFusion2D


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, _, _ = get_video_loaders()

videos, labels = next(iter(train_loader))

print("Input shape:", videos.shape)

model = EarlyFusion2D(
    num_classes=10,
    num_frames=10
).to(device)

videos = videos.to(device)

outputs = model(videos)

print("Output shape:", outputs.shape)