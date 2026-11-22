import tkinter as tk
from PIL import Image, ImageTk
from pathlib import Path

root = tk.Tk()
root.title("SmartAir Display Test")

image_path = Path("dataset/images/video1_frame_00001.jpg")

image = Image.open(image_path)

image.thumbnail((900, 600))

photo = ImageTk.PhotoImage(image)

label = tk.Label(root, image=photo)
label.pack()

root.mainloop()

