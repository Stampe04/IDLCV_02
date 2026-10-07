import torch
import torch.nn as nn

from dataloader import get_video_loaders
from train import train_model
from late_fusion_2D.late_fusion_2D import LateFusion2D


def main():

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # Data
    train_loader, val_loader, test_loader = get_video_loaders(
        batch_size=16,
        num_workers=4
    )

    # Model
    model = LateFusion2D(
        num_classes=10,
        num_frames=10
    ).to(device)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    # Train
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=20,
        save_path="late_fusion_2D/results/late_fusion_metrics.csv"
    )


if __name__ == "__main__":
    main()