import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True)

def get_eye_region(landmarks, indices, img_w, img_h):
    return np.array([(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in indices])

def get_eye_gaze_direction(landmarks, w, h):
    # Left eye: indices around iris and lid
    left_eye_indices = [33, 133, 160, 159, 158, 144, 153, 154, 155, 173]  # Includes iris and outer eye
    right_eye_indices = [362, 263, 387, 386, 385, 373, 380, 381, 382, 362]

    left_eye_pts = get_eye_region(landmarks, left_eye_indices, w, h)
    right_eye_pts = get_eye_region(landmarks, right_eye_indices, w, h)

    # Use center of iris landmarks to estimate pupil location
    left_iris_center = np.mean(get_eye_region(landmarks, [468], w, h), axis=0)
    right_iris_center = np.mean(get_eye_region(landmarks, [473], w, h), axis=0)

    # Bounding box for each eye
    left_xs = [p[0] for p in left_eye_pts]
    right_xs = [p[0] for p in right_eye_pts]

    # Normalize iris location inside eye
    left_ratio = (left_iris_center[0] - min(left_xs)) / (max(left_xs) - min(left_xs))
    right_ratio = (right_iris_center[0] - min(right_xs)) / (max(right_xs) - min(right_xs))

    avg_ratio = (left_ratio + right_ratio) / 2

    if avg_ratio < 0.35:
        return "looking left"
    elif avg_ratio > 0.65:
        return "looking right"
    else:
        return "looking center"

def estimate_pose_and_gaze(frame):
    h, w = frame.shape[:2]
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return "no face detected"

    landmarks = results.multi_face_landmarks[0].landmark

    # ---- Head Pose Estimation ----
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ], dtype='double')

    image_points = np.array([
        (landmarks[1].x * w, landmarks[1].y * h),    # Nose tip
        (landmarks[152].x * w, landmarks[152].y * h),# Chin
        (landmarks[263].x * w, landmarks[263].y * h),# Left eye left corner
        (landmarks[33].x * w, landmarks[33].y * h),  # Right eye right corner
        (landmarks[287].x * w, landmarks[287].y * h),# Left mouth corner
        (landmarks[57].x * w, landmarks[57].y * h)   # Right mouth corner
    ], dtype='double')

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype='double')

    dist_coeffs = np.zeros((4,1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs)

    if not success:
        return "pose estimation failed"

    rmat, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rmat, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    yaw = euler_angles[1][0]
    pitch = euler_angles[0][0]

    if abs(yaw) < 15 and abs(pitch) < 15:
        head_direction = "head forward"
    elif yaw < -15:
        head_direction = "head right"
    elif yaw > 15:
        head_direction = "head left"
    elif pitch < -10:
        head_direction = "head up"
    elif pitch > 10:
        head_direction = "head down"
    else:
        head_direction = "head away"

    # ---- Eye Gaze Tracking ----
    eye_gaze = get_eye_gaze_direction(landmarks, w, h)

    return f"{head_direction}, {eye_gaze}"
