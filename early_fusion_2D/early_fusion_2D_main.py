from seed import set_seed
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from dataloader import get_video_loaders
from train import train_model
from evaluate import evaluate_model
from seed import set_seed

from early_fusion_2D.early_fusion_2D_model import EarlyFusion2D


def main():

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)


    # Experiment settings

    seeds = [42, 0, 2026, 1, 123]

    num_epochs = 100

    batch_size = 16

    num_workers = 4

    learning_rate = 1e-3

    # Dataset
    #
    # Original dataset used for Section 4.1

    root_dir = "/dtu/datasets1/02516/ufc10"


    # --------------------------------------------------
    # Results directory
    # --------------------------------------------------

    results_dir = Path(
        "early_fusion_2D/results"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # Store final test results from each seed
    all_results = []

    # Train once for each seed

    for seed in seeds:

        print("\n" + "=" * 60)
        print(f"Starting seed {seed}")
        print("=" * 60)

        # Set seed

        set_seed(seed)

        # DataLoaders

        train_loader, val_loader, test_loader = (
            get_video_loaders(
                root_dir=root_dir,
                batch_size=batch_size,
                num_workers=num_workers,
                seed=seed
            )
        )


        # ----------------------------------------------
        # Create a NEW model for this seed
        # ----------------------------------------------

        model = EarlyFusion2D(
            num_classes=10,
            num_frames=10
        ).to(device)


        # ----------------------------------------------
        # Loss function
        # ----------------------------------------------

        criterion = nn.CrossEntropyLoss()


        # ----------------------------------------------
        # Create a NEW optimizer for this seed
        # ----------------------------------------------

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate
        )


        # ----------------------------------------------
        # Create a NEW scheduler for this seed
        # ----------------------------------------------

        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5
            )
        )


        # ----------------------------------------------
        # Separate output folder for this seed
        # ----------------------------------------------

        seed_dir = (
            results_dir /
            f"seed_{seed}"
        )

        seed_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        training_metrics_path = (
            seed_dir /
            "training_metrics.csv"
        )

        best_model_path = (
            seed_dir /
            "best_model.pt"
        )

        test_metrics_path = (
            seed_dir /
            "test_metrics.csv"
        )


        # ----------------------------------------------
        # Train
        # ----------------------------------------------

        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            save_path=training_metrics_path,
            best_model_path=best_model_path,
            scheduler=scheduler
        )


        # ----------------------------------------------
        # Load the best validation checkpoint
        # ----------------------------------------------

        model.load_state_dict(
            torch.load(
                best_model_path,
                map_location=device
            )
        )


        # ----------------------------------------------
        # Evaluate on the test set
        # ----------------------------------------------

        results, _, _ = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device,
            save_path=test_metrics_path
        )


        # ----------------------------------------------
        # Add seed to results
        # ----------------------------------------------

        results["seed"] = seed

        all_results.append(results)


        print(
            f"\nFinished seed {seed}"
        )


    # --------------------------------------------------
    # Combine results from all five seeds
    # --------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )


    # Put seed first
    results_df = results_df[
        [
            "seed",
            "accuracy",
            "precision",
            "recall",
            "f1"
        ]
    ]


    # --------------------------------------------------
    # Save combined results
    # --------------------------------------------------

    combined_results_path = (
        results_dir /
        "all_seed_test_metrics.csv"
    )

    results_df.to_csv(
        combined_results_path,
        index=False
    )


    # --------------------------------------------------
    # Print final results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("All seeds completed")
    print("=" * 60)

    print(results_df)

    print(
        f"\nCombined results saved to: "
        f"{combined_results_path}"
    )


if __name__ == "__main__":
    main()