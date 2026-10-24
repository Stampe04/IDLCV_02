import os
import sys
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if __package__:
    from .resnet_3D import BasicBlock, ResNet_3D, inflate_resnet18_weights
else:
    from resnet_3D.resnet_3D import BasicBlock, ResNet_3D, inflate_resnet18_weights

from dataloader import get_video_loaders
from evaluate import evaluate_model
from train import train_model


def get_env_int(name, default):
    return int(os.environ.get(name, default))


def get_env_float(name, default):
    return float(os.environ.get(name, default))


def get_env_layers(name, default):
    value = os.environ.get(name, default)
    layers = [int(depth.strip()) for depth in value.split(",")]
    if len(layers) != 4 or any(depth <= 0 for depth in layers):
        raise ValueError(
            f"{name} must contain four positive integers separated by commas"
        )
    return layers


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    data_root = os.environ.get(
        "RESNET_3D_DATA_ROOT",
        "/dtu/datasets1/02516/ufc10"
    )
    batch_size = get_env_int("RESNET_3D_BATCH_SIZE", 16)
    num_workers = get_env_int("RESNET_3D_NUM_WORKERS", 4)
    num_epochs = get_env_int("RESNET_3D_NUM_EPOCHS", 100)
    layers = get_env_layers("RESNET_3D_LAYERS", "1,1,1,1")
    base_channels = get_env_int("RESNET_3D_BASE_CHANNELS", 16)
    dropout = get_env_float("RESNET_3D_DROPOUT", 0.1)
    learning_rate = get_env_float("RESNET_3D_LEARNING_RATE", 1e-3)
    scheduler_factor = get_env_float(
        "RESNET_3D_SCHEDULER_FACTOR",
        0.5
    )
    scheduler_patience = get_env_int(
        "RESNET_3D_SCHEDULER_PATIENCE",
        5
    )
    results_dir = Path(os.environ.get(
        "RESNET_3D_RESULTS_DIR",
        str(Path(__file__).resolve().parent / "results")
    ))
    results_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = get_video_loaders(
        root_dir=data_root,
        batch_size=batch_size,
        num_workers=num_workers
    )


    from torchvision.models import resnet18, ResNet18_Weights

    # Load pretrained 2D ResNet-18
    model_2d = resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    # Create your 3D ResNet-18
    model = ResNet_3D(
        block=BasicBlock,
        layers=layers,
        num_classes=10,
        in_channels=3,
        base_channels=base_channels,
        dropout=dropout
    )

    # Transfer pretrained weights
    model = inflate_resnet18_weights(model, model_2d)

    model = model.to(device)


    num_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
    print(f"Trainable parameters: {num_params:,}")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience
    )

    metrics_path = results_dir / "resnet_3D_metrics.csv"
    best_model_path = results_dir / "best_model.pt"
    test_metrics_path = results_dir / "test_metrics.csv"

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        save_path=str(metrics_path),
        best_model_path=str(best_model_path),
        scheduler=scheduler
    )

    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=device
        )
    )

    evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        save_path=str(test_metrics_path)
    )


if __name__ == "__main__":
    main()