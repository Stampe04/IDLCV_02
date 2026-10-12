import torch
import pandas as pd


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs=100,
    save_path="metrics.csv",
    best_model_path="best_model.pt",
    scheduler=None
):
    # Store metrics
    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []

    learning_rates = []

    best_val_accuracy = 0.0

    # Training loop
    for epoch in range(num_epochs):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            predicted = outputs.argmax(dim=1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)


        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total

        # Validation
        model.eval()

        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for inputs, labels in val_loader:

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = criterion(outputs, labels)

                val_running_loss += loss.item()

                predicted = outputs.argmax(dim=1)

                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)


        val_loss = val_running_loss / len(val_loader)
        val_accuracy = val_correct / val_total

        # Learning rate scheduler
        if scheduler is not None:
            scheduler.step(val_loss)


        current_lr = optimizer.param_groups[0]["lr"]

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        learning_rates.append(current_lr)

        # Save best model
        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            torch.save(
                model.state_dict(),
                best_model_path
            )

            print(
                f"New best model saved "
                f"(val acc: {best_val_accuracy:.4f})"
            )

        # Save metrics
        metrics = pd.DataFrame({
            "epoch": range(1, len(train_losses) + 1),
            "train_loss": train_losses,
            "val_loss": val_losses,
            "train_accuracy": train_accuracies,
            "val_accuracy": val_accuracies,
            "learning_rate": learning_rates
        })

        metrics.to_csv(
            save_path,
            index=False
        )

        # Print epoch results
        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_accuracy:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_accuracy:.4f} | "
            f"LR: {current_lr:.6f}"
        )


    print("Training finished.")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Metrics saved to: {save_path}")

    return metrics