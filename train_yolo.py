
from ultralytics import YOLO
import torch

print("SMARTAIR YOLO TRAINING STARTED")

# Check GPU
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Load pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train using airport dataset and NVIDIA GPU
results = model.train(
    data="yolo_dataset/data.yaml",
    epochs=30,
    imgsz=640,
    batch=2,
    device=0,
    workers=0,
    project="runs/smartair",
    name="airport_model",
    exist_ok=True
)

print("SMARTAIR TRAINING COMPLETED!")
