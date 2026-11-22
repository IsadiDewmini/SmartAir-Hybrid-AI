
from ultralytics import YOLO
from pathlib import Path
import csv

print("SMARTAIR COUNT COMPARISON STARTED")

# Load trained YOLO model
model = YOLO(
    r"C:\Users\HP\runs\detect\runs\smartair\airport_model\weights\best.pt"
)

# Dataset folders
images_folder = Path("dataset/images")
labels_folder = Path("dataset/labels")

results_data = []

total_error = 0
total_images = 0
exact_matches = 0

image_files = sorted(images_folder.glob("*.jpg"))

for image_path in image_files:

    label_path = labels_folder / (image_path.stem + ".txt")

    if not label_path.exists():
        print("Label Missing:", image_path.name)
        continue

    # Ground Truth count
    with open(label_path, "r") as file:
        ground_truth = sum(
            1 for line in file if line.strip()
        )

    # YOLO prediction
    prediction = model.predict(
        source=str(image_path),
        conf=0.25,
        verbose=False
    )

    predicted_count = len(prediction[0].boxes)

    # Calculate error
    error = abs(ground_truth - predicted_count)

    total_error += error
    total_images += 1

    if error == 0:
        exact_matches += 1

    results_data.append([
        image_path.name,
        ground_truth,
        predicted_count,
        error
    ])

    print(
        image_path.name,
        "| Actual:", ground_truth,
        "| Predicted:", predicted_count,
        "| Error:", error
    )

# Save comparison results
with open("smartair_comparison.csv", "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Image",
        "Ground_Truth",
        "YOLO_Prediction",
        "Count_Error"
    ])

    writer.writerows(results_data)

# Final evaluation
if total_images > 0:

    mae = total_error / total_images

    print("\n==============================")
    print("SMARTAIR EVALUATION RESULTS")
    print("==============================")

    print("Total Images:", total_images)
    print("Exact Count Matches:", exact_matches)
    print("Mean Absolute Error (MAE):", round(mae, 2))

    print("\nComparison CSV Saved!")
    print("smartair_comparison.csv")

else:
    print("No images were evaluated.")
