
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from smartair_cnn import SmartAirCNN

print("SMARTAIR VIDEO CROWD MONITORING STARTED")

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_device = 0 if torch.cuda.is_available() else "cpu"

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# Find airport videos
video_folder = Path("dataset/vedios")

video_files = []

for extension in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
    video_files.extend(video_folder.glob(extension))

video_files = sorted(video_files)

if not video_files:
    raise FileNotFoundError(
        "No videos found inside dataset/vedios folder!"
    )

print("\nAvailable videos:")

for index, video in enumerate(video_files):
    print(index + 1, "-", video.name)

# Select first video
video_path = str(video_files[0])

print("\nSelected video:", video_path)


# Load YOLO model
yolo_model = YOLO(
    "runs/detect/runs/smartair/airport_model/weights/best.pt"
)

# Load CNN model
cnn_model = SmartAirCNN().to(device)

cnn_model.load_state_dict(
    torch.load(
        "smartair_cnn.pth",
        map_location=device,
        weights_only=True
    )
)

cnn_model.eval()

print("\nYOLO model loaded!")
print("SmartAir CNN loaded!")


# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("Cannot open the selected video!")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 25

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("\nVideo FPS:", fps)
print("Video resolution:", width, "x", height)
print("Total frames:", total_frames)


# Create output video
output_path = "smartair_video_result.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    raise RuntimeError("Cannot create output video!")


# Process video frames
frame_number = 0

print("\nSMARTAIR VIDEO PROCESSING STARTED")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # YOLO person detection
    results = yolo_model.predict(
        source=frame,
        conf=0.10,
        iou=0.45,
        device=yolo_device,
        verbose=False
    )

    yolo_count = sum(
        1 for box in results[0].boxes
        if int(box.cls[0].item()) == 0
    )

    # CNN crowd count
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    resized = cv2.resize(rgb_frame, (320, 320))

    normalized = resized.astype(np.float32) / 255.0

    input_data = np.transpose(normalized, (2, 0, 1))

    input_tensor = torch.tensor(
        input_data,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        cnn_prediction = cnn_model(input_tensor).item()

    # Crowd classification
    if cnn_prediction <= 5:
        crowd_level = "LOW"

    elif cnn_prediction <= 10:
        crowd_level = "MEDIUM"

    else:
        crowd_level = "HIGH"

    # Draw YOLO boxes
    output_frame = results[0].plot()

    # Add SmartAir information
    cv2.putText(
        output_frame,
        f"YOLO Count: {yolo_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        output_frame,
        f"CNN Estimate: {cnn_prediction:.2f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.putText(
        output_frame,
        f"Crowd Level: {crowd_level}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # Save processed frame
    writer.write(output_frame)

    if frame_number % 30 == 0:
        print(
            f"Processed {frame_number}/{total_frames} frames"
        )


# Release resources
cap.release()
writer.release()

print("\n================================")
print("SMARTAIR VIDEO PROCESSING COMPLETED!")
print("Total frames processed:", frame_number)
print("Output video:", output_path)
print("================================")
