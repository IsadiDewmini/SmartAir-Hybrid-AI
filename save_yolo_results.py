import cv2
import pandas as pd

from pathlib import Path
from ultralytics import YOLO

# Load YOLO model
print("Loading YOLO11n Model...")

model = YOLO("yolo11n.pt")

# Dataset folders
image_folder = Path("dataset/images")
output_folder = Path("dataset/yolo_results")

output_folder.mkdir(parents=True, exist_ok=True)

# Get all images
image_files = sorted(image_folder.glob("*.jpg"))

results_data = []

print("\nTotal Images:", len(image_files))
print("\nStarting YOLO Detection...\n")

# Process images
for image_path in image_files:

    results = model.predict(
        source=str(image_path),
        classes=[0],
        conf=0.25,
        verbose=False
    )

    result = results[0]

    # Count detected people
    person_count = len(result.boxes)

    # Calculate average confidence
    if person_count > 0:
        confidences = result.boxes.conf.cpu().numpy()
        average_confidence = float(confidences.mean())
    else:
        average_confidence = 0.0

    # Save annotated image
    annotated_image = result.plot()

    output_path = output_folder / image_path.name

    cv2.imwrite(str(output_path), annotated_image)

    # Store results
    results_data.append({
        "Image_Name": image_path.name,
        "People_Count": person_count,
        "Average_Confidence": round(average_confidence, 4)
    })

    print(
        image_path.name,
        "| People:", person_count,
        "| Confidence:", round(average_confidence, 4)
    )

# Create DataFrame
df = pd.DataFrame(results_data)

# Save CSV
csv_path = Path("dataset/yolo_counts.csv")

df.to_csv(csv_path, index=False)

# Summary
print("\n================================")
print("YOLO DETECTION COMPLETED!")
print("================================")

print("Total Images:", len(df))
print("Total Detected People:", df["People_Count"].sum())
print("Average People Count:", round(df["People_Count"].mean(), 2))
print("Average Confidence:", round(df["Average_Confidence"].mean(), 4))

print("\nCSV Saved:", csv_path)

print("\n================================")
