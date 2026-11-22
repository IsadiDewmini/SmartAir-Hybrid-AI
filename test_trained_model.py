
from ultralytics import YOLO

model = YOLO(
    "runs/detect/runs/smartair/airport_model/weights/best.pt"
)

image_path = "dataset/images/video1_frame_00001.jpg"

for iou_value in [0.3, 0.45, 0.6, 0.7]:

    results = model.predict(
        source=image_path,
        conf=0.18,
        iou=iou_value,
        device=0,
        save=False,
        verbose=False
    )

    count = sum(
        1 for box in results[0].boxes
        if int(box.cls[0]) == 0
    )

    print(f"IoU: {iou_value} | People: {count}")

print("IOU TEST COMPLETED!")

