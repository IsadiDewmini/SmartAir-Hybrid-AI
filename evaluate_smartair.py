
from ultralytics import YOLO
from pathlib import Path

print("SMARTAIR CONFIDENCE OPTIMIZATION")

model = YOLO(
    "runs/detect/runs/smartair/airport_model/weights/best.pt"
)

image_folder = Path("yolo_dataset/images/val")
label_folder = Path("yolo_dataset/labels/val")

confidence_values = [0.05, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25]

# Store actual counts
dataset = []

for image_path in sorted(image_folder.iterdir()):

    if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    label_path = label_folder / (image_path.stem + ".txt")

    if not label_path.exists():
        continue

    with open(label_path, "r") as file:
        actual = sum(1 for line in file if line.strip())

    dataset.append((image_path, actual))

best_conf = None
best_mae = float("inf")

for conf in confidence_values:

    total_error = 0

    for image_path, actual in dataset:

        results = model.predict(
            source=str(image_path),
            conf=conf,
            iou=0.45,
            device=0,
            save=False,
            verbose=False
        )

        predicted = sum(
            1 for box in results[0].boxes
            if int(box.cls[0]) == 0
        )

        total_error += abs(actual - predicted)

    mae = total_error / len(dataset)

    print(f"Confidence: {conf:.2f} | MAE: {mae:.2f}")

    if mae < best_mae:
        best_mae = mae
        best_conf = conf

print("\nBEST CONFIDENCE:", best_conf)
print("BEST VALIDATION MAE:", round(best_mae, 2))

print("OPTIMIZATION COMPLETED!")

