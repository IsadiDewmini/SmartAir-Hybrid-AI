
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

print("SMARTAIR CNN DATASET PREPARATION")

image_folder = Path("dataset/images")
label_folder = Path("dataset/labels")

X = []
y = []
records = []

for image_path in sorted(image_folder.glob("*.jpg")):

    label_path = label_folder / (image_path.stem + ".txt")

    if not label_path.exists():
        print("Missing label:", image_path.name)
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        print("Cannot read:", image_path.name)
        continue

    # Resize image
    image = cv2.resize(image, (320, 320))

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize pixel values
    image = image.astype(np.float32) / 255.0

    # Count annotated persons
    lines = [
        line for line in label_path.read_text().splitlines()
        if line.strip()
    ]

    person_count = 0

    for line in lines:
        class_id = int(line.split()[0])

        if class_id == 0:
            person_count += 1

    X.append(image)
    y.append(person_count)

    records.append({
        "image": image_path.name,
        "person_count": person_count
    })

# Convert to NumPy arrays
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# Save dataset
np.save("cnn_images.npy", X)
np.save("cnn_counts.npy", y)

pd.DataFrame(records).to_csv(
    "cnn_dataset.csv",
    index=False
)

print("\nDATASET PREPARATION COMPLETED")
print("Images shape:", X.shape)
print("Counts shape:", y.shape)
print("Total people:", int(y.sum()))
print("Average people per image:", round(float(y.mean()), 2))
print("Maximum people in one image:", int(y.max()))
print("Minimum people in one image:", int(y.min()))
