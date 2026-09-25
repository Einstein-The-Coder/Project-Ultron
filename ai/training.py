import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import TensorDataset, DataLoader

from .model import UltronModel


def train(
    epochs=30,
    batch_size=64,
    learning_rate=0.001
):

    print("Loading training data...")

    digits = load_digits()

    X = digits.data
    y = digits.target


    # --------------------------------------------------------
    # Split dataset
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    # --------------------------------------------------------
    # Standardize
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)


    # --------------------------------------------------------
    # Convert to PyTorch
    # --------------------------------------------------------

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_train = torch.tensor(
        y_train,
        dtype=torch.long
    )


    dataset = TensorDataset(
        X_train,
        y_train
    )


    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )


    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = UltronModel()


    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print()
    print("===================================")
    print("        ULTRON TRAINING")
    print("===================================")

    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print()


    for epoch in range(epochs):

        model.train()

        total_loss = 0
        correct = 0
        total = 0


        for inputs, labels in loader:

            # Clear previous gradients
            optimizer.zero_grad()


            # Forward pass
            outputs = model(inputs)


            # Calculate loss
            loss = criterion(
                outputs,
                labels
            )


            # Backpropagation
            loss.backward()


            # Update weights
            optimizer.step()


            # Statistics
            total_loss += loss.item()


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            correct += (
                predictions == labels
            ).sum().item()


            total += labels.size(0)


        average_loss = (
            total_loss / len(loader)
        )

        accuracy = correct / total


        print(
            f"Epoch {epoch + 1:02d}/{epochs} "
            f"| Loss: {average_loss:.4f} "
            f"| Accuracy: {accuracy * 100:.2f}%"
        )


    # --------------------------------------------------------
    # Create model directory
    # --------------------------------------------------------

    os.makedirs(
        "models",
        exist_ok=True
    )


    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = "models/ultron_digits.pth"

    torch.save(
        model.state_dict(),
        model_path
    )


    # --------------------------------------------------------
    # Save scaler
    # --------------------------------------------------------

    scaler_path = "models/ultron_scaler.pkl"

    joblib.dump(
        scaler,
        scaler_path
    )


    print()
    print("===================================")
    print("       TRAINING COMPLETE")
    print("===================================")

    print(
        f"Model saved to: {model_path}"
    )

    print(
        f"Scaler saved to: {scaler_path}"
    )


    return model, scaler