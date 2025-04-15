import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2

class ProctorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Proctoring System")

        # Video Frame
        self.video_label = tk.Label(root)
        self.video_label.pack()

        # Status Frame
        self.status_frame = ttk.LabelFrame(root, text="Proctoring Status")
        self.status_frame.pack(fill="x")

        self.status_texts = {
            "gadget": tk.StringVar(value="No gadgets detected"),
            "person": tk.StringVar(value="1 person detected"),
            "lip": tk.StringVar(value="No lip movement"),
            "pose": tk.StringVar(value="Looking forward")
        }

        for key, var in self.status_texts.items():
            ttk.Label(self.status_frame, textvariable=var, foreground="green").pack(anchor="w")

    def update_video(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def update_status(self, key, message, alert=False):
        self.status_texts[key].set(message)
        color = "red" if alert else "green"
        self.status_frame.children[list(self.status_frame.children)[list(self.status_texts).index(key)]].configure(foreground=color)
