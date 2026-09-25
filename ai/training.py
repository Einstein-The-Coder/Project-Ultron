import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import TensorDataset, DataLoader

from .model import UltronModel


def train():

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


    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)


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
        batch_size=64,
        shuffle=True
    )


    model = UltronModel()


    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )


    epochs = 30


    print("Starting training...")


    for epoch in range(epochs):

        model.train()

        total_loss = 0


        for inputs, labels in loader:

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()


        average_loss = (
            total_loss / len(loader)
        )


        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Loss: {average_loss:.4f}"
        )


    print("Training complete.")

    return model, scaler