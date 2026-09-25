import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# 1. Load the dataset
# ============================================================

digits = load_digits()

X = digits.data
y = digits.target

print("Dataset shape:", X.shape)
print("Number of classes:", len(np.unique(y)))


# ============================================================
# 2. Put the data into a Pandas DataFrame
# ============================================================

df = pd.DataFrame(X)
df["label"] = y

print("\nFirst five rows:")
print(df.head())


# ============================================================
# 3. Visualize some digits
# ============================================================

fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap="gray")
    ax.set_title(f"Label: {y[i]}")
    ax.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# 4. Train/test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 5. Standardize the input data
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ============================================================
# 6. Convert NumPy arrays into PyTorch tensors
# ============================================================

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


# ============================================================
# 7. Create PyTorch datasets and dataloaders
# ============================================================

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# ============================================================
# 8. Define the neural network
# ============================================================

class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(64, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.network(x)


model = NeuralNetwork()

print("\nNeural network:")
print(model)


# ============================================================
# 9. Loss function and optimizer
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# 10. Train the neural network
# ============================================================

epochs = 30

training_losses = []
training_accuracies = []

for epoch in range(epochs):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{epochs}"
    )

    for inputs, labels in progress:

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        progress.set_postfix(
            loss=loss.item()
        )

    average_loss = total_loss / len(train_loader)

    accuracy = correct / total

    training_losses.append(average_loss)
    training_accuracies.append(accuracy)

    print(
        f"Loss: {average_loss:.4f} | "
        f"Accuracy: {accuracy:.4f}"
    )


# ============================================================
# 11. Evaluate the network
# ============================================================

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for inputs, labels in test_loader:

        outputs = model(inputs)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


# ============================================================
# 12. Accuracy
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print("\nFinal test accuracy:")
print(f"{accuracy * 100:.2f}%")


# ============================================================
# 13. Classification report
# ============================================================

print("\nClassification report:")

print(
    classification_report(
        all_labels,
        all_predictions
    )
)


# ============================================================
# 14. Confusion matrix
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

plt.figure(figsize=(9, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Neural Network Confusion Matrix")

plt.show()


# ============================================================
# 15. Plot training loss
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    training_losses,
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")

plt.grid()

plt.show()


# ============================================================
# 16. Plot training accuracy
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    training_accuracies,
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")

plt.grid()

plt.show()


# ============================================================
# 17. Display some predictions
# ============================================================

model.eval()

sample_indices = np.random.choice(
    len(X_test),
    10,
    replace=False
)

fig, axes = plt.subplots(
    2,
    5,
    figsize=(12, 5)
)

for ax, index in zip(
    axes.flat,
    sample_indices
):

    image = X_test[index].numpy()

    output = model(
        X_test[index].unsqueeze(0)
    )

    prediction = torch.argmax(
        output,
        dim=1
    ).item()

    # Undo standardization for visualization
    image = scaler.inverse_transform(
        image.reshape(1, -1)
    ).reshape(8, 8)

    ax.imshow(
        image,
        cmap="gray"
    )

    ax.set_title(
        f"Predicted: {prediction}\n"
        f"Actual: {y_test[index].item()}"
    )

    ax.axis("off")

plt.tight_layout()
plt.show()

