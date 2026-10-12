import torch
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(
    model,
    test_loader,
    device,
    save_path=None
):
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for inputs, labels in test_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())


    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )


    print(f"Test accuracy:  {accuracy:.4f}")
    print(f"Macro precision: {precision:.4f}")
    print(f"Macro recall:    {recall:.4f}")
    print(f"Macro F1:        {f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            all_labels,
            all_predictions,
            zero_division=0
        )
    )

    print("\nConfusion matrix:")
    print(
        confusion_matrix(
            all_labels,
            all_predictions
        )
    )


    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


    if save_path is not None:

        pd.DataFrame(
            [results]
        ).to_csv(
            save_path,
            index=False
        )


    return results, all_labels, all_predictions