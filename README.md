# FaceAttend — Biometric Attendance System

A desktop application that uses face recognition to automate attendance tracking. Built with Python, it uses deep learning models (SCRFD + ArcFace) running on ONNX Runtime to detect and identify faces in real-time through a webcam feed.

## How It Works

1. **Register** — Admin captures an employee's face through the webcam. The system extracts a 512-dimensional face embedding using ArcFace and stores it in a local SQLite database alongside user details.

2. **Scan** — When someone stands in front of the camera, the system detects their face using SCRFD, generates their embedding, and compares it against all registered users using cosine similarity. If a match is found above the threshold (default 40%), their attendance is logged with a timestamp.

3. **Dashboard** — View attendance records by date, see who checked in and when, and export reports to CSV.

## Tech Stack

- **Face Detection**: SCRFD (via InsightFace's `buffalo_l` model pack)
- **Face Recognition**: ArcFace — generates 512-D face embeddings
- **Inference**: ONNX Runtime (CPU, with optional GPU support)
- **GUI**: CustomTkinter (modern dark-themed desktop UI)
- **Camera**: OpenCV with threaded capture
- **Database**: SQLite
- **Language**: Python 3.10+

## Setup

### Prerequisites
- Python 3.10 or higher
- A working webcam
- ~500MB disk space (for model files downloaded on first run)

### Installation

```bash
# clone the repo
git clone https://github.com/Avneesh11905/attendance_system-.git
cd attendance_system-

# create virtual environment
python -m venv venv

# activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

### Running

```bash
python -m app.main
```

On the first run, the InsightFace `buffalo_l` model pack (~300MB) will be downloaded automatically. This only happens once.

## Usage

### Registering a User
1. Click **Register** in the sidebar
2. Position the person's face in the camera
3. Click **Capture Face** — the system will detect and highlight the face
4. Fill in Name, Employee ID, and Department
5. Click **Register User**

### Scanning Attendance
1. Click **Scan** in the sidebar
2. Auto-scan is on by default — just stand in front of the camera
3. The system will match your face and log attendance automatically
4. A green checkmark means attendance was recorded; orange means already checked in today

### Viewing Records
1. Click **Dashboard** in the sidebar
2. Navigate between dates using the arrow buttons
3. Click **Export CSV** to save records to a file

### Settings
- **Similarity Threshold** — controls how strict face matching is (lower = more lenient)
- **Camera** — switch between connected cameras

## Project Structure

```
├── app/
│   ├── main.py              # entry point
│   ├── config.py             # settings and constants
│   ├── core/
│   │   ├── face_engine.py    # SCRFD + ArcFace inference wrapper
│   │   ├── camera.py         # threaded camera capture
│   │   └── database.py       # SQLite operations
│   ├── ui/
│   │   ├── app_window.py     # main window with sidebar
│   │   ├── scan_frame.py     # live scanning screen
│   │   ├── register_frame.py # user registration screen
│   │   ├── dashboard_frame.py# attendance records view
│   │   └── settings_frame.py # app settings
│   └── utils/
│       └── helpers.py        # image conversion, embedding utils
├── data/                     # SQLite database (auto-created)
├── face_models/              # ONNX models (auto-downloaded)
└── requirements.txt
```

## How the Recognition Pipeline Works

```
Camera Frame
    ↓
SCRFD Face Detection (det_10g.onnx)
    ↓  bounding box + 5 facial landmarks
Face Alignment (affine transform using landmarks)
    ↓  normalized 112×112 face crop
ArcFace Embedding (w600k_r50.onnx)
    ↓  512-dimensional feature vector
Cosine Similarity vs. registered embeddings
    ↓  score > threshold?
Log Attendance to SQLite
```

## Notes

- Face embeddings are stored as binary blobs in SQLite — no images are saved
- Each user can only check in once per day (duplicate prevention)
- Camera runs in a separate thread so the UI stays responsive
- Inference runs in a background thread to avoid frame drops
- The InsightFace `buffalo_l` models are for non-commercial/research use. See [InsightFace license](https://github.com/deepinsight/insightface/tree/master/python-package) for details.
