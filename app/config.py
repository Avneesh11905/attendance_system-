import os

# paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "attendance.db")
MODEL_DIR = os.path.join(BASE_DIR, "face_models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# face engine
MODEL_NAME = "buffalo_l"
SIMILARITY_THRESHOLD = 0.4
DET_SIZE = (640, 640)

# camera
DEFAULT_CAMERA_INDEX = 0
CAMERA_FPS = 30
FRAME_SKIP = 3  # run inference every Nth frame

# attendance rules
SCAN_COOLDOWN_SECONDS = 30
LATE_THRESHOLD_HOUR = 9  # 9 AM

# ui
APP_TITLE = "FaceAttend — Biometric Attendance System"
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 700
CAMERA_DISPLAY_WIDTH = 640
CAMERA_DISPLAY_HEIGHT = 480
