"""
Threaded camera capture using OpenCV.
Runs in a background thread so the GUI doesn't freeze.
"""

import cv2
import threading
import numpy as np
from typing import Optional


class CameraManager:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self):
        """Open camera and start the capture thread."""
        if self._running:
            return

        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Can't open camera {self.camera_index}. "
                "Is it plugged in or being used by something else?"
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        """Keeps reading frames in the background. Only the latest frame is kept."""
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame

    def get_frame(self) -> Optional[np.ndarray]:
        """Return a copy of the latest frame, or None if nothing captured yet."""
        with self._lock:
            if self._frame is not None:
                return self._frame.copy()
            return None

    def stop(self):
        """Stop capture thread and release camera."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()
            self._cap = None
        self._frame = None

    def switch_camera(self, new_index: int):
        """Stop current camera and start a different one."""
        self.stop()
        self.camera_index = new_index
        self.start()

    @staticmethod
    def list_cameras(max_check: int = 5) -> list[int]:
        """Try opening camera indices 0..max_check and return which ones work."""
        available = []
        for i in range(max_check):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available
