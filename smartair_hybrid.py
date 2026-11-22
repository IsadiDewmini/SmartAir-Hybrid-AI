
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from smartair_cnn import SmartAirCNN

print("SMARTAIR HYBRID AI STARTED")

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# Load YOLO model
yolo_model = YOLO(
    "runs/detect/runs/smartair/airport_model/weights/best.pt"
)


# Load our CNN model
cnn_model = SmartAirCNN().to(device)

cnn_model.load_state_dict(
    torch.load(
        "smartair_cnn.pth",
        map_location=device,
        weights_only=True
    )
)

cnn_model.eval()

print("YOLO model loaded!")
print("SmartAir CNN loaded!")


# Select test image
image_path = "dataset/images/video1_frame_00001.jpg"

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Test image not found!")


# YOLO person detection
yolo_results = yolo_model.predict(
    source=image_path,
    conf=0.10,
    iou=0.45,
    device=0 if torch.cuda.is_available() else "cpu",
    verbose=False
)

yolo_count = sum(
    int(box.cls[0]) == 0
    for box in yolo_results[0].boxes
)


# CNN crowd count prediction
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

resized = cv2.resize(rgb_image, (320, 320))

normalized = resized.astype(np.float32) / 255.0

input_data = np.transpose(normalized, (2, 0, 1))

input_tensor = torch.tensor(
    input_data,
    dtype=torch.float32
).unsqueeze(0).to(device)

with torch.no_grad():
    cnn_prediction = cnn_model(input_tensor).item()


# Display results
print("\n========== SMARTAIR RESULTS ==========")

print("YOLO Detected People:", yolo_count)

print("CNN Estimated People:", round(cnn_prediction, 2))

print("CNN Rounded Count:", round(cnn_prediction))


# Crowd Level Classification

if cnn_prediction <= 5:
    crowd_level = "LOW"

elif cnn_prediction <= 10:
    crowd_level = "MEDIUM"

else:
    crowd_level = "HIGH"


print("Crowd Level:", crowd_level)


print("======================================")

# Draw YOLO bounding boxes
annotated_image = yolo_results[0].plot()

cv2.putText(
    annotated_image,
    f"YOLO Count: {yolo_count}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

cv2.putText(
    annotated_image,
    f"CNN Estimate: {cnn_prediction:.2f}",
    (20, 80),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 255),
    2
)

# Display Crowd Level on Image

cv2.putText(
    annotated_image,
    f"Crowd Level: {crowd_level}",
    (20, 120),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255),
    2
)

cv2.imwrite("smartair_hybrid_result.jpg", annotated_image)

print("\nResult image saved: smartair_hybrid_result.jpg")
print("SMARTAIR HYBRID AI COMPLETED!")
