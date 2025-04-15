import torch
import cv2
import os
from datetime import datetime
from modules.logger import log_event

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Create snapshot folder
SNAPSHOT_FOLDER = "snapshots_person"
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)

def count_people(frame):
    results = model(frame)
    df = results.pandas().xyxy[0]
    persons = df[df['name'] == 'person']
    person_count = len(persons)

    warning = False
    snapshot_path = ""

    if person_count != 1:
        warning = True
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_path = f"{SNAPSHOT_FOLDER}/person_violation_{timestamp}.jpg"
        cv2.imwrite(snapshot_path, frame)

        # Log offense with consistent format
        reason = "No person detected" if person_count == 0 else "Multiple people detected"
        log_event(reason, snapshot_path)

    return person_count, warning, snapshot_path
