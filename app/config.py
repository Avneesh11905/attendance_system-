import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "attendance.db")
MODEL_DIR = os.path.join(BASE_DIR, "face_models")

# Ensure required directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# Face Engine Settings
# ──────────────────────────────────────────────
MODEL_NAME = "buffalo_l"              # InsightFace model pack
SIMILARITY_THRESHOLD = 0.4           # Cosine similarity cutoff for a match
DET_SIZE = (640, 640)                # Detection input resolution

# ──────────────────────────────────────────────
# Camera Settings
# ──────────────────────────────────────────────
DEFAULT_CAMERA_INDEX = 0
CAMERA_FPS = 30
FRAME_SKIP = 3                       # Run inference every Nth frame

# ──────────────────────────────────────────────
# Attendance Rules
# ──────────────────────────────────────────────
SCAN_COOLDOWN_SECONDS = 30           # Same person can't re-trigger within this window
LATE_THRESHOLD_HOUR = 9             # Hour after which check-in counts as "Late"

# ──────────────────────────────────────────────
# UI Constants
# ──────────────────────────────────────────────
APP_TITLE = "FaceAttend — Biometric Attendance System"
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 700
CAMERA_DISPLAY_WIDTH = 640
CAMERA_DISPLAY_HEIGHT = 480
