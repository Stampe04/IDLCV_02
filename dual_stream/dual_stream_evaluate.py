import torch

from evaluate_dual import evaluate_model
from dataloader import get_twostream_loaders
from dual_stream.dual_stream_model import DualStreamNetwork


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    _, _, test_loader = get_twostream_loaders(
        batch_size=16,
        num_workers=4
    )

    model = DualStreamNetwork(
        num_classes=10,
        num_frames=10
    ).to(device)

    model.load_state_dict(
        torch.load(
            "dual_stream/results/best_model.pt",
            map_location=device
        )
    )

    evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        save_path=(
            "dual_stream/results/"
            "test_metrics_noleak.csv"
        )
    )


if __name__ == "__main__":
    main()