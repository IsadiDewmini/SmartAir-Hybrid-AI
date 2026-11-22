from pathlib import Path
import random
import shutil

# Dataset paths
image_folder = Path("dataset/images")
label_folder = Path("dataset/labels")

output_folder = Path("yolo_dataset")

# Create folders
for folder in [
    output_folder / "images/train",
    output_folder / "images/val",
    output_folder / "labels/train",
    output_folder / "labels/val"
]:
    folder.mkdir(parents=True, exist_ok=True)

# Get images that have label files
image_files = []

for image_path in image_folder.glob("*.jpg"):
    label_path = label_folder / f"{image_path.stem}.txt"

    if label_path.exists():
        image_files.append(image_path)

# Shuffle dataset
random.seed(42)
random.shuffle(image_files)

# Split 80% train and 20% validation
split_index = int(len(image_files) * 0.8)

train_images = image_files[:split_index]
val_images = image_files[split_index:]

# Copy files
def copy_files(images, split):
    for image_path in images:
        label_path = label_folder / f"{image_path.stem}.txt"

        shutil.copy2(
            image_path,
            output_folder / f"images/{split}" / image_path.name
        )

        shutil.copy2(
            label_path,
            output_folder / f"labels/{split}" / label_path.name
        )

copy_files(train_images, "train")
copy_files(val_images, "val")

print("Dataset Preparation Completed!")
print("Training Images:", len(train_images))
print("Validation Images:", len(val_images))
print("Total Images:", len(image_files))