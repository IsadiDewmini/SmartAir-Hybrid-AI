
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from smartair_cnn import SmartAirCNN

print("SMARTAIR CNN EVALUATION STARTED")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
X = np.load("cnn_images.npy")
y = np.load("cnn_counts.npy")

X = np.transpose(X, (0, 3, 1, 2))

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

dataset = TensorDataset(X_tensor, y_tensor)

# Recreate the same validation split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)

_, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

val_loader = DataLoader(val_dataset, batch_size=4)

# Load trained model
model = SmartAirCNN().to(device)

model.load_state_dict(
    torch.load(
        "smartair_cnn.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()

actual_counts = []
predicted_counts = []

with torch.no_grad():

    for images, counts in val_loader:

        images = images.to(device)

        predictions = model(images)

        actual_counts.extend(counts.numpy())
        predicted_counts.extend(predictions.cpu().numpy())

# Calculate evaluation metrics
actual = np.array(actual_counts)
predicted = np.array(predicted_counts)

errors = np.abs(actual - predicted)

mae = np.mean(errors)
rmse = np.sqrt(np.mean((actual - predicted) ** 2))

print("\nVALIDATION RESULTS")

for i in range(len(actual)):
    print(
        f"Image {i+1} | "
        f"Actual: {actual[i]:.0f} | "
        f"Predicted: {predicted[i]:.2f} | "
        f"Error: {errors[i]:.2f}"
    )

print("\nCNN PERFORMANCE")
print("Validation Images:", len(actual))
print("MAE:", round(float(mae), 3))
print("RMSE:", round(float(rmse), 3))

print("\nEVALUATION COMPLETED")

