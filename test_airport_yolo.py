from ultralytics import YOLO
import cv2
from pathlib import Path

model = YOLO("yolo11n.pt")

image_folder = Path("dataset/images")
output_folder = Path("dataset/yolo_results")

output_folder.mkdir(parents=True, exist_ok=True)

image_files = sorted(image_folder.glob("*.jpg"))

print("Total Images:", len(image_files))

for image_path in image_files:

    results = model.predict(
        source=str(image_path),
        classes=[0],
        conf=0.25,
        verbose=False
    )

    result = results[0]

    person_count = len(result.boxes)

    annotated_image = result.plot()

    output_path = output_folder / image_path.name

    cv2.imwrite(str(output_path), annotated_image)

    print(image_path.name, "-> People:", person_count)

print("\nYOLO AIRPORT TEST COMPLETED!")