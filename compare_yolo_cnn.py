
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from smartair_cnn import SmartAirCNN

print("SMARTAIR YOLO vs CNN COMPARISON")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained models
yolo = YOLO("runs/detect/runs/smartair/airport_model/weights/best.pt")

cnn = SmartAirCNN().to(device)
cnn.load_state_dict(
    torch.load("smartair_cnn.pth", map_location=device, weights_only=True)
)
cnn.eval()

# Same 12 YOLO validation images
image_folder = Path("yolo_dataset/images/val")
label_folder = Path("dataset/labels")

yolo_errors = []
cnn_errors = []

for image_path in sorted(image_folder.glob("*.jpg")):

    label_path = label_folder / (image_path.stem + ".txt")

    if not label_path.exists():
        print("Missing label:", image_path.name)
        continue

    # Actual count
    lines = [
        line for line in label_path.read_text().splitlines()
        if line.strip()
    ]

    actual = sum(int(line.split()[0]) == 0 for line in lines)

    # YOLO prediction
    results = yolo.predict(
        source=str(image_path),
        conf=0.10,
        iou=0.45,
        device=0,
        verbose=False
    )

    yolo_count = sum(
        int(box.cls[0]) == 0
        for box in results[0].boxes
    )

    # CNN prediction
    image = cv2.imread(str(image_path))
    image = cv2.resize(image, (320, 320))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0

    image = np.transpose(image, (2, 0, 1))
    image_tensor = torch.tensor(
        image, dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        cnn_count = cnn(image_tensor).item()

    yolo_error = abs(actual - yolo_count)
    cnn_error = abs(actual - cnn_count)

    yolo_errors.append(yolo_error)
    cnn_errors.append(cnn_error)

    print(
        f"{image_path.name} | Actual: {actual} | "
        f"YOLO: {yolo_count} | CNN: {cnn_count:.2f}"
    )

print("\nFINAL COMPARISON")

print("Images tested:", len(yolo_errors))
print("YOLO MAE:", round(float(np.mean(yolo_errors)), 3))
print("CNN MAE:", round(float(np.mean(cnn_errors)), 3))

print("\nCOMPARISON COMPLETED")

