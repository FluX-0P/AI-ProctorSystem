from datetime import datetime

LOG_FILE = "activity_log.txt"

def log_event(offense_name, snapshot_path="", transcript=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"{offense_name}, {snapshot_path}, {timestamp}"
    if transcript:
        log_line += f", Transcript: {transcript}"
    with open(LOG_FILE, "a") as f:
        f.write(log_line + "\n")
