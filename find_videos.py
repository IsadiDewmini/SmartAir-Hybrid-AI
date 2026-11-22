from pathlib import Path

project = Path.cwd()

print("Current Project:", project)

downloads = Path.home() / "Downloads"

print("\nSearching for MP4 videos...")

for file in downloads.rglob("*.mp4"):
    print("Video Found:", file)

print("\nSearch Completed!")