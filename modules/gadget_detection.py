import torch
import cv2
import os
from datetime import datetime
from modules.logger import log_event

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Create snapshot folder if not exists
SNAPSHOT_FOLDER = "snapshots_gadget"
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)

# Track repeated offences
gadget_offense_count = 0

def detect_gadgets(frame):
    global gadget_offense_count

    results = model(frame)
    df = results.pandas().xyxy[0]
    gadgets = df[df['name'].isin(['cell phone', 'laptop', 'tv'])]

    repeated_offense = False
    snapshot_path = ""

    if not gadgets.empty:
        gadget_offense_count += 1
        
        if gadget_offense_count >= 3:  # Take snapshot only on repeated offense
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{SNAPSHOT_FOLDER}/gadget_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            snapshot_path = filename
            log_event("Repeated Gadget Use Detected", snapshot_path)
            repeated_offense = True

    return gadgets, repeated_offense, snapshot_path