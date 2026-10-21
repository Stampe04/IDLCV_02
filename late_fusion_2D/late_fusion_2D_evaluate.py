import torch

from dataloader import get_video_loaders
from evaluate import evaluate_model
from late_fusion_2D.late_fusion_2D import LateFusion2D


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    _, _, test_loader = get_video_loaders(
        batch_size=16,
        num_workers=4
    )

    model = LateFusion2D(
        num_classes=10,
        num_frames=10
    ).to(device)

    model.load_state_dict(
        torch.load(
            "late_fusion_2D/results/best_model.pt",
            map_location=device
        )
    )

    evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        save_path=(
            "late_fusion_2D/results/"
            "test_metrics_noleak.csv"
        )
    )


if __name__ == "__main__":
    main()