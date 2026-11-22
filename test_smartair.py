from ultralytics import YOLO
from pathlib import Path

# Load our trained model
model = YOLO(
    r"C:\Users\HP\runs\detect\runs\smartair\airport_model\weights\best.pt"
)

# Test image folder
source = "dataset/images"

# Run detection
results = model.predict(
    source=source,
    conf=0.25,
    save=True
)

print("SMARTAIR DETECTION COMPLETED!")

for result in results:
    print(
        result.path,
        "People Detected:",
        len(result.boxes)
    )