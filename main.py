import cv2
import threading
import time
import os
from gui import ProctorGUI
import tkinter as tk
from tkinter import messagebox
from modules import gadget_detection, person_detection, lip_tracking, head_pose
from modules.audio_transcript import record_and_transcribe
from modules.lip_tracking import detect_lip_activity_only
from modules.logger import log_event

# Audio monitoring function
def monitor_audio():
    global audio_active, latest_transcript
    while True:
        text = record_and_transcribe()
        if text:
            audio_active = True
            latest_transcript = text
        else:
            audio_active = False

# Main app function
def run_app():
    cap = cv2.VideoCapture(0)
    root = tk.Tk()
    gui = ProctorGUI(root)

    # Start audio monitor thread
    threading.Thread(target=monitor_audio, daemon=True).start()

    # Tracking variables
    lip_active = False
    last_talking_time = 0
    talking_logged = False
    gaze_off_counter = 0
    gaze_threshold = 5  # Adjust based on time or frames
    repeated_gaze_offense_count = 0  # Counter for repeated gaze offenses

    def update():
        nonlocal lip_active, last_talking_time, talking_logged, gaze_off_counter, repeated_gaze_offense_count

        ret, frame = cap.read()
        frame = cv2.flip(frame,1)

        if not ret:
            root.after(10, update)
            return

        gui.update_video(frame)

        # Gadget Detection
        gadgets, repeated_offense, snapshot_path = gadget_detection.detect_gadgets(frame)
        if not gadgets.empty:
            if repeated_offense:
                gui.update_status("gadget", "Repeated Gadget Use Detected!", alert=True)
                messagebox.showwarning("Cheating Detected", "Repeated gadget use has been detected!")
                log_event("Gadget Use (Repeated)", snapshot_path)
            else:
                gui.update_status("gadget", "Gadget Detected!", alert=True)
                log_event("Gadget Use", snapshot_path)
        else:
            gui.update_status("gadget", "No gadgets")

        # Person Detection
        persons, person_warning, person_path = person_detection.count_people(frame)
        if person_warning:
            gui.update_status("person", f"{persons} person(s) detected", alert=True)
            messagebox.showwarning("Cheating Detected", f"{persons} person(s) detected!")
        else:
            gui.update_status("person", f"{persons} person(s) detected")

        # Talking Detection
        global audio_active, latest_transcript
        is_talking, lip_variation = detect_lip_activity_only(frame)  # Updated function call
        current_time = time.time()

        if is_talking and audio_active:
            last_talking_time = current_time
            if not talking_logged:
                gui.update_status("lip", "Talking Detected!", alert=True)
                snapshot_path = f"snapshots_mouth/talking_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(snapshot_path, frame)
                log_event("Talking Detected", snapshot_path, f"Lip Variation: {lip_variation:.2f}, Transcript: {latest_transcript}")
                talking_logged = True
        elif current_time - last_talking_time > 5:
            gui.update_status("lip", "Not talking")
            talking_logged = False


        # Head Pose and Gaze Detection with sustained gaze tracking logic
        pose_gaze = head_pose.estimate_pose_and_gaze(frame)

        # Create snapshot folder if not exists
        SNAPSHOT_FOLDER = "snapshots_head_pose"
        os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)
        
        if "forward" not in pose_gaze and "center" not in pose_gaze:  # Gaze away from screen detected.
            gaze_off_counter += 1
            
            if gaze_off_counter == gaze_threshold:  # Threshold reached for sustained gaze away.
                snapshot_path = f"snapshots_head_pose/gaze_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(snapshot_path, frame)
                log_event("Head/Eye Away for Too Long", snapshot_path)
                messagebox.showwarning("Cheating Detected", "Looking away for too long!")
                
                repeated_gaze_offense_count += 1
                
                if repeated_gaze_offense_count >= 3:  # Repeated offense detected.
                    snapshot_path_repeated = f"head_pose_snapshot/repeated_gaze_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(snapshot_path_repeated, frame)
                    log_event("Repeated Head/Eye Away", snapshot_path_repeated)
                    messagebox.showwarning("Cheating Detected", "Repeated gaze away detected!")
        
        else:  # Reset counter when gaze returns to screen.
            gaze_off_counter = 0

        gui.update_status("pose", f"{pose_gaze}", alert=(gaze_off_counter >= gaze_threshold))

        root.after(10, update)

    update()
    root.mainloop()
    cap.release()

if __name__ == "__main__":
    # Initialize shared variables
    audio_active = False
    latest_transcript = ""

    run_app()
