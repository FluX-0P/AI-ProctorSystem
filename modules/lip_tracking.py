import mediapipe as mp
import cv2
import math
import os
from datetime import datetime
from collections import deque
from modules.logger import log_event

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

UPPER_LIP_INDEX = 13
LOWER_LIP_INDEX = 14

MOUTH_SNAPSHOT_FOLDER = "snapshots_mouth"
os.makedirs(MOUTH_SNAPSHOT_FOLDER, exist_ok=True)

# For tracking movement
lip_distances = deque(maxlen=15)  # About 0.5 sec at 30 fps
MOUTH_MOVEMENT_VARIATION_THRESHOLD = 2.5  # Adjust as needed

last_log_time = None  # Track the last log time for snapshots/logs.
LOG_INTERVAL = 5  # Minimum interval (in seconds) between logs/snapshots.

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)

def get_lip_distance(face_landmarks, frame_height):
    """Calculate the distance between the upper and lower lips."""
    upper = face_landmarks.landmark[UPPER_LIP_INDEX]
    lower = face_landmarks.landmark[LOWER_LIP_INDEX]
    return calculate_distance(upper, lower) * frame_height

def calculate_lip_variation():
    """Calculate the variation in lip distances."""
    return max(lip_distances) - min(lip_distances)

def should_log_event():
    """Check if enough time has passed since the last log event."""
    global last_log_time
    current_time = datetime.now()
    if last_log_time is None or (current_time - last_log_time).total_seconds() > LOG_INTERVAL:
        last_log_time = current_time
        return True
    return False

def save_mouth_snapshot(frame):
    """Save a snapshot of mouth activity."""
    snapshot_path = os.path.join(MOUTH_SNAPSHOT_FOLDER, f"mouth_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
    cv2.imwrite(snapshot_path, frame)
    log_event("Talking Detected", snapshot_path)

def detect_lip_activity_only(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, _, _ = frame.shape
            distance = get_lip_distance(face_landmarks, h)
            lip_distances.append(distance)

            if len(lip_distances) >= 5:
                variation = calculate_lip_variation()
                return variation > MOUTH_MOVEMENT_VARIATION_THRESHOLD, variation  # Return tuple
    return False, 0  # Return False with zero variation if no activity detected
