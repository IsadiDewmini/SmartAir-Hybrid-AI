
from ultralytics import YOLO

print("Loading SmartAir YOLO Model...")

# Load pre-trained YOLO11 Nano model
model = YOLO("yolo11n.pt")

print("YOLO Model Loaded Successfully!")

# Test with sample image
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    conf=0.25,
    imgsz=640,
    device="cpu",
    save=True
)

# Count detected people
for result in results:
    person_count = sum(
        1 for cls in result.boxes.cls
        if int(cls) == 0
    )

    print("Detected People:", person_count)

print("YOLO Testing Completed!")
