
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from pathlib import Path

# ==================================
# 1. FOLDER SETTINGS
# ==================================

image_folder = Path("dataset/images")
label_folder = Path("dataset/labels")

label_folder.mkdir(parents=True, exist_ok=True)

image_files = sorted(image_folder.glob("*.jpg"))

if len(image_files) < 2:
    print("Not enough images!")
    exit()

print("Total Images:", len(image_files))


# ==================================
# 2. VARIABLES
# ==================================

current_index = 1

annotations = {}

photo = None
scale = 1.0

start_x = None
start_y = None

preview_box = None

original_width = 0
original_height = 0


# ==================================
# 3. CREATE WINDOW
# ==================================

root = tk.Tk()
root.title("SmartAir Annotation Tool")
root.geometry("1150x780")
root.configure(bg="#202020")


title = tk.Label(
    root,
    text="SmartAir - Airport Person Annotation",
    font=("Arial", 18, "bold"),
    fg="white",
    bg="#202020"
)
title.pack(pady=10)


info_label = tk.Label(
    root,
    text="",
    font=("Arial", 11),
    fg="white",
    bg="#202020"
)
info_label.pack(pady=5)


canvas = tk.Canvas(
    root,
    bg="white",
    highlightthickness=0
)
canvas.pack(pady=10)


# ==================================
# 4. DISPLAY IMAGE
# ==================================

def show_image():

    global photo, scale
    global original_width, original_height

    image_path = image_files[current_index]

    image = Image.open(image_path).convert("RGB")

    original_width, original_height = image.size

    scale = min(
        1050 / original_width,
        580 / original_height,
        1.0
    )

    display_width = int(original_width * scale)
    display_height = int(original_height * scale)

    image = image.resize(
        (display_width, display_height),
        Image.Resampling.LANCZOS
    )

    photo = ImageTk.PhotoImage(image)

    canvas.config(
        width=display_width,
        height=display_height
    )

    canvas.delete("all")

    canvas.create_image(
        0, 0,
        anchor="nw",
        image=photo
    )

    draw_saved_boxes()

    info_label.config(
        text=(
            f"Image {current_index + 1}/{len(image_files)} | "
            f"{image_path.name} | "
            f"People: {len(annotations.get(current_index, []))}"
        )
    )

    root.update_idletasks()

    print("Displaying:", image_path.name)


# ==================================
# 5. DRAW EXISTING BOXES
# ==================================

def draw_saved_boxes():

    for box in annotations.get(current_index, []):

        x1, y1, x2, y2 = box

        canvas.create_rectangle(
            x1 * scale,
            y1 * scale,
            x2 * scale,
            y2 * scale,
            outline="lime",
            width=2
        )


# ==================================
# 6. START BOX
# ==================================

def start_box(event):

    global start_x, start_y

    start_x = event.x
    start_y = event.y


# ==================================
# 7. PREVIEW BOX
# ==================================

def draw_preview(event):

    global preview_box

    if start_x is None:
        return

    if preview_box is not None:
        canvas.delete(preview_box)

    preview_box = canvas.create_rectangle(
        start_x,
        start_y,
        event.x,
        event.y,
        outline="yellow",
        width=2
    )


# ==================================
# 8. FINISH BOX
# ==================================

def finish_box(event):

    global start_x, start_y, preview_box

    if start_x is None:
        return

    x1 = min(start_x, event.x)
    y1 = min(start_y, event.y)

    x2 = max(start_x, event.x)
    y2 = max(start_y, event.y)

    if x2 - x1 > 5 and y2 - y1 > 5:

        box = [
            x1 / scale,
            y1 / scale,
            x2 / scale,
            y2 / scale
        ]

        annotations.setdefault(
            current_index, []
        ).append(box)

    start_x = None
    start_y = None
    preview_box = None

    show_image()


# ==================================
# 9. UNDO
# ==================================

def undo_box():

    if annotations.get(current_index):

        annotations[current_index].pop()

        show_image()

    else:

        messagebox.showinfo(
            "Undo",
            "No boxes available!"
        )


# ==================================
# 10. SAVE YOLO LABEL
# ==================================

def save_current():

    image_path = image_files[current_index]

    label_path = label_folder / (
        image_path.stem + ".txt"
    )

    with open(label_path, "w") as file:

        for box in annotations.get(current_index, []):

            x1, y1, x2, y2 = box

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            box_width = x2 - x1
            box_height = y2 - y1

            center_x /= original_width
            center_y /= original_height

            box_width /= original_width
            box_height /= original_height

            file.write(
                f"0 {center_x:.6f} "
                f"{center_y:.6f} "
                f"{box_width:.6f} "
                f"{box_height:.6f}\n"
            )

    print(
        "Saved:",
        label_path.name,
        "| People:",
        len(annotations.get(current_index, []))
    )


def save_button():

    save_current()

    messagebox.showinfo(
        "Saved",
        "YOLO annotations saved successfully!"
    )


# ==================================
# 11. NEXT IMAGE
# ==================================

def next_image():

    global current_index

    save_current()

    if current_index < len(image_files) - 1:

        current_index += 1
        show_image()

    else:

        messagebox.showinfo(
            "Info",
            "Last image reached!"
        )


# ==================================
# 12. PREVIOUS IMAGE
# ==================================

def previous_image():

    global current_index

    save_current()

    if current_index > 1:

        current_index -= 1
        show_image()

    else:

        messagebox.showinfo(
            "Info",
            "This is the first annotation image!"
        )


# ==================================
# 13. BUTTONS
# ==================================

button_frame = tk.Frame(
    root,
    bg="#202020"
)

button_frame.pack(pady=10)


tk.Button(
    button_frame,
    text="Previous",
    width=12,
    command=previous_image
).pack(side="left", padx=5)


tk.Button(
    button_frame,
    text="Undo",
    width=12,
    command=undo_box
).pack(side="left", padx=5)


tk.Button(
    button_frame,
    text="Save",
    width=12,
    command=save_button
).pack(side="left", padx=5)


tk.Button(
    button_frame,
    text="Next",
    width=12,
    command=next_image
).pack(side="left", padx=5)


# ==================================
# 14. MOUSE EVENTS
# ==================================

canvas.bind("<ButtonPress-1>", start_box)
canvas.bind("<B1-Motion>", draw_preview)
canvas.bind("<ButtonRelease-1>", finish_box)


# ==================================
# 15. START TOOL
# ==================================

print("\nSMARTAIR ANNOTATION TOOL STARTED")

show_image()

root.mainloop()
from pathlib import Path

label_folder = Path("dataset/labels")
label_files = list(label_folder.glob("*.txt"))

print("Total Label Files:", len(label_files))

for file in label_files[:5]:
    print(file.name, ":", file.read_text().strip()[:100])
