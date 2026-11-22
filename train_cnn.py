
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from smartair_cnn import SmartAirCNN

print("SMARTAIR CNN TRAINING STARTED")

# Select GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Load dataset
X = np.load("cnn_images.npy")
y = np.load("cnn_counts.npy")

# Convert NHWC to NCHW
X = np.transpose(X, (0, 3, 1, 2))

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

dataset = TensorDataset(X_tensor, y_tensor)

# Split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4
)

# Create model
model = SmartAirCNN().to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Training
epochs = 100

for epoch in range(epochs):

    model.train()
    train_loss = 0

    for images, counts in train_loader:

        images = images.to(device)
        counts = counts.to(device)

        optimizer.zero_grad()

        predictions = model(images)

        loss = criterion(predictions, counts)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0

    with torch.no_grad():

        for images, counts in val_loader:

            images = images.to(device)
            counts = counts.to(device)

            predictions = model(images)

            loss = criterion(predictions, counts)

            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Validation Loss: {val_loss:.4f}"
    )

# Save trained model
torch.save(model.state_dict(), "smartair_cnn.pth")

print("\nSMARTAIR CNN TRAINING COMPLETED!")
print("Model saved as smartair_cnn.pth")
