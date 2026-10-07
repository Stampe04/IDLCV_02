import os
import sys
import tempfile

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from resnet_3D import BasicBlock, ResNet_3D
from train import train_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Match the shape produced by FrameVideoDataset: (N, C, frames, H, W).
    videos = torch.randn(4, 3, 10, 64, 64)
    labels = torch.randint(0, 10, (4,))
    loader = DataLoader(TensorDataset(videos, labels), batch_size=2)

    model = ResNet_3D(
        BasicBlock,
        [2, 2, 2, 2],
        num_classes=10,
        in_channels=3
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as metrics_file:
        metrics_path = metrics_file.name

    try:
        metrics = train_model(
            model,
            loader,
            loader,
            criterion,
            optimizer,
            device=device,
            num_epochs=1,
            save_path=metrics_path
        )
    finally:
        os.unlink(metrics_path)

    assert len(metrics) == 1
    print(f"Pipeline proof of concept passed on {device}.")


if __name__ == "__main__":
    main()