from ultralytics import YOLO

model = YOLO(
    "runs/detect/runs/smartair/airport_model/weights/best.pt"
)

image_path = "dataset/images/video1_frame_00001.jpg"

results = model.predict(
    source=image_path,
    conf=0.18,
    device=0,
    save=False
)

for result in results:
    print("\nSMARTAIR DETECTION DETAILS")
    print("--------------------------")

    print("Total detections:", len(result.boxes))

    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = float(box.conf[0])

        print(
            f"Person {i+1}: "
            f"Confidence={confidence:.3f}, "
            f"Box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})"
        )

print("\nINSPECTION COMPLETED!")