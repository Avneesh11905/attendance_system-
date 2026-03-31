"""
Entry point — loads models, initializes everything, and launches the GUI.
"""

import sys
from app.config import DEFAULT_CAMERA_INDEX
from app.core.face_engine import FaceEngine
from app.core.camera import CameraManager
from app.core.database import Database
from app.ui.app_window import AppWindow


def main():
    print("Initializing face engine (this may download models on first run)...")
    engine = FaceEngine()
    try:
        engine.load_models()
    except Exception as e:
        print(f"Failed to load face models: {e}")
        print("Make sure you have insightface and onnxruntime installed.")
        sys.exit(1)

    print("Setting up database...")
    db = Database()

    print("Starting camera...")
    camera = CameraManager(DEFAULT_CAMERA_INDEX)

    print("Launching application...")
    app = AppWindow(engine, camera, db)
    app.mainloop()


if __name__ == "__main__":
    main()
