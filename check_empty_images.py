
import cv2
from pathlib import Path

image_folder = Path("dataset/images")

images_to_check = [
    "video1_frame_00000.jpg",
    "video2_frame_00000.jpg",
    "video3_frame_00000.jpg",
    "video4_frame_00000.jpg",
    "video5_frame_00000.jpg",
    "video6_frame_00000.jpg"
]

for image_name in images_to_check:

    image_path = image_folder / image_name
    image = cv2.imread(str(image_path))

    if image is None:
        print("Cannot load:", image_name)
        continue

    cv2.imshow(image_name, image)
    print("Viewing:", image_name)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("IMAGE CHECK COMPLETED")
