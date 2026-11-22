import cv2
from pathlib import Path

image_folder = Path("dataset/images")
image_files = sorted(image_folder.glob("*.jpg"))

print("Total Images:", len(image_files))
print("\nImage Details:\n")

for image_path in image_files[:10]:
    image = cv2.imread(str(image_path))

    if image is not None:
        height, width, channels = image.shape
        print(
            image_path.name,
            "| Width:", width,
            "| Height:", height,
            "| Channels:", channels
        )
    else:
        print("Cannot Read:", image_path.name)

print("\nDataset Inspection Completed!")