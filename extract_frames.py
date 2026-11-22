import cv2
from pathlib import Path

# Airport dataset location
dataset_path = Path.home() / "Downloads" / (
    "Metaverse Airport Crowd Anomaly Detection Dataset"
) / "Metaverse Airport Crowd Anomaly Detection Dataset" / (
    "Metaverse airport surveillance videos frames Dataset"
)

# Find only airport videos
video_files = sorted(dataset_path.rglob("*.mp4"))

# Output folders
image_folder = Path("dataset/images")
image_folder.mkdir(parents=True, exist_ok=True)

print("Airport Videos Found:", len(video_files))

total_saved = 0

# Process each video
for video_index, video_path in enumerate(video_files, start=1):

    print("\nProcessing:", video_path.name)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Cannot open:", video_path.name)
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("FPS:", fps)
    print("Total Frames:", frame_count)

    # Extract one frame every 5 seconds
    frame_interval = max(1, round(fps * 5))

    frame_number = 0
    saved_count = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        if frame_number % frame_interval == 0:

            filename = (
                f"video{video_index}_frame_{saved_count:05d}.jpg"
            )

            save_path = image_folder / filename

            if cv2.imwrite(str(save_path), frame):
                saved_count += 1
                total_saved += 1

        frame_number += 1

    cap.release()

    print("Extracted:", saved_count, "images")

print("\n================================")
print("FRAME EXTRACTION COMPLETED!")
print("Total Videos:", len(video_files))
print("Total Images:", total_saved)
print("================================")
