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
    # ==========================================
    # DEVICE
    # ==========================================

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print()
        print("===================================")
        print("       NVIDIA GPU DETECTED")
        print("===================================")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
        print()
    else:
        device = torch.device("cpu")

        print()
        print("===================================")
        print("          CPU MODE")
        print("===================================")
        print("CUDA GPU not available.")
        print()

    # ==========================================
    # LOAD DATA
    # ==========================================

    print("Loading training data...")

    digits = load_digits()

    X = digits.data
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ==========================================
    # SCALE DATA
    # ==========================================

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ==========================================
    # MOVE DATA TO PYTORCH
    # ==========================================

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_train = torch.tensor(
        y_train,
        dtype=torch.long
    )

    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    y_test = torch.tensor(
        y_test,
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

    # ==========================================
    # CREATE MODEL
    # ==========================================

    model = UltronModel().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # ==========================================
    # TRAINING INFO
    # ==========================================

    print()
    print("===================================")
    print("        ULTRON TRAINING")
    print("===================================")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print()

    # ==========================================
    # TRAIN
    # ==========================================

    for epoch in range(epochs):

        model.train()

        total_loss = 0
        correct = 0
        total = 0

        for inputs, labels in loader:

            # Move batch to GPU
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        average_loss = total_loss / len(loader)

        accuracy = correct / total

        print(
            f"Epoch {epoch + 1:02d}/{epochs} "
            f"| Loss: {average_loss:.4f} "
            f"| Train Accuracy: {accuracy * 100:.2f}%"
        )

    # ==========================================
    # EVALUATION
    # ==========================================

    print()
    print("===================================")
    print("          ULTRON EVALUATION")
    print("===================================")

    model.eval()

    with torch.no_grad():

        X_test_gpu = X_test.to(device)
        y_test_gpu = y_test.to(device)

        outputs = model(X_test_gpu)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct = (
            predictions == y_test_gpu
        ).sum().item()

        test_accuracy = (
            correct / len(y_test)
        )

    print(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    # ==========================================
    # SAVE
    # ==========================================

    os.makedirs(
        "models",
        exist_ok=True
    )

    model_path = (
        "models/ultron_digits.pth"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    scaler_path = (
        "models/ultron_scaler.pkl"
    )

    joblib.dump(
        scaler,
        scaler_path
    )

    print()
    print("===================================")
    print("        TRAINING COMPLETE")
    print("===================================")
    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(
        f"Final test accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    return model, scaler