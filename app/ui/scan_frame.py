"""
Live scan screen — shows camera feed, detects faces, matches against
registered users, and logs attendance automatically.
"""

import time
import threading
import customtkinter as ctk
from app.config import (
    CAMERA_DISPLAY_WIDTH, CAMERA_DISPLAY_HEIGHT,
    SIMILARITY_THRESHOLD, SCAN_COOLDOWN_SECONDS,
)
from app.utils.helpers import frame_to_ctkimage, draw_face_box


class ScanFrame(ctk.CTkFrame):
    def __init__(self, parent, face_engine, camera, database):
        super().__init__(parent, fg_color="transparent")

        self.face_engine = face_engine
        self.camera = camera
        self.db = database

        self._running = True
        self._auto_scan = True
        self._frame_count = 0
        self._last_scan_times = {}  # user_id -> timestamp (cooldown tracking)
        self._last_result = None

        self._build_ui()

        # start camera if not already running
        if not self.camera.is_running:
            try:
                self.camera.start()
            except RuntimeError as e:
                self._set_status(str(e), "red")

        self._update_feed()

    def _build_ui(self):
        # two columns: camera on left, status on right
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        # left side — camera
        cam_frame = ctk.CTkFrame(self, corner_radius=12, fg_color="#1a1a2e")
        cam_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        header = ctk.CTkLabel(
            cam_frame, text="Live Camera Feed",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        header.pack(pady=(15, 10))

        self.camera_label = ctk.CTkLabel(cam_frame, text="Starting camera...")
        self.camera_label.pack(padx=15, pady=(0, 15))

        # auto-scan toggle
        toggle_frame = ctk.CTkFrame(cam_frame, fg_color="transparent")
        toggle_frame.pack(pady=(0, 15))

        self.auto_scan_switch = ctk.CTkSwitch(
            toggle_frame, text="Auto Scan",
            command=self._toggle_auto_scan,
            onvalue=True, offvalue=False,
        )
        self.auto_scan_switch.select()  # on by default
        self.auto_scan_switch.pack(side="left", padx=10)

        self.scan_btn = ctk.CTkButton(
            toggle_frame, text="Manual Scan",
            command=self._manual_scan,
            fg_color="#e94560", hover_color="#c81e45",
            width=120,
        )
        self.scan_btn.pack(side="left", padx=10)

        # right side — status panel
        status_frame = ctk.CTkFrame(self, corner_radius=12, fg_color="#1a1a2e")
        status_frame.grid(row=0, column=1, sticky="nsew")

        status_header = ctk.CTkLabel(
            status_frame, text="Scan Result",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        status_header.pack(pady=(15, 20))

        # big status icon/text
        self.status_icon = ctk.CTkLabel(
            status_frame, text="📷",
            font=ctk.CTkFont(size=60),
        )
        self.status_icon.pack(pady=(10, 5))

        self.status_text = ctk.CTkLabel(
            status_frame, text="Position your face\nin the camera",
            font=ctk.CTkFont(size=16),
            text_color="#aaaaaa",
            justify="center",
        )
        self.status_text.pack(pady=(5, 15))

        # details (shown after a match)
        self.detail_frame = ctk.CTkFrame(status_frame, fg_color="#16213e", corner_radius=10)
        self.detail_frame.pack(fill="x", padx=20, pady=10)

        self.name_label = ctk.CTkLabel(
            self.detail_frame, text="",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.name_label.pack(pady=(12, 2))

        self.dept_label = ctk.CTkLabel(
            self.detail_frame, text="",
            font=ctk.CTkFont(size=13), text_color="#aaaaaa",
        )
        self.dept_label.pack(pady=(0, 2))

        self.conf_label = ctk.CTkLabel(
            self.detail_frame, text="",
            font=ctk.CTkFont(size=12), text_color="#888888",
        )
        self.conf_label.pack(pady=(0, 5))

        self.time_label = ctk.CTkLabel(
            self.detail_frame, text="",
            font=ctk.CTkFont(size=12), text_color="#888888",
        )
        self.time_label.pack(pady=(0, 12))

        # hide details initially
        self.detail_frame.pack_forget()

        # stats at bottom
        stats_frame = ctk.CTkFrame(status_frame, fg_color="transparent")
        stats_frame.pack(side="bottom", fill="x", padx=20, pady=15)

        self.registered_label = ctk.CTkLabel(
            stats_frame, text="Registered: 0",
            font=ctk.CTkFont(size=12), text_color="#888888",
        )
        self.registered_label.pack()

        self.today_label = ctk.CTkLabel(
            stats_frame, text="Today: 0 check-ins",
            font=ctk.CTkFont(size=12), text_color="#888888",
        )
        self.today_label.pack()

        self._update_stats()

    def _update_feed(self):
        """Called every ~33ms to refresh camera display and optionally run inference."""
        if not self._running:
            return

        frame = self.camera.get_frame()
        if frame is not None:
            self._frame_count += 1

            # run face detection on every few frames if auto-scan is on
            if self._auto_scan and self._frame_count % 5 == 0:
                # run inference in a thread so we don't lag the UI
                frame_copy = frame.copy()
                threading.Thread(
                    target=self._run_inference, args=(frame_copy,), daemon=True
                ).start()

            # draw box if we have a recent result
            if self._last_result:
                result_type, data = self._last_result
                if result_type == "match":
                    user, score, bbox = data
                    label = f"{user.name} ({score:.0%})"
                    draw_face_box(frame, bbox, label, color=(0, 255, 0))
                elif result_type == "unknown":
                    bbox = data
                    draw_face_box(frame, bbox, "Unknown", color=(0, 0, 255))

            img = frame_to_ctkimage(frame, CAMERA_DISPLAY_WIDTH, CAMERA_DISPLAY_HEIGHT)
            self.camera_label.configure(image=img, text="")
            self.camera_label._image = img  # keep reference

        self.after(33, self._update_feed)

    def _run_inference(self, frame):
        """Detect + recognize in background thread."""
        if not self.face_engine.is_loaded:
            return

        faces = self.face_engine.detect_and_embed(frame)
        if not faces:
            self._last_result = None
            self.after(0, lambda: self._set_status("No face detected", "#aaaaaa"))
            return

        face = faces[0]  # just use the first/biggest face
        users = self.db.get_all_users()

        if not users:
            self._last_result = ("unknown", face.bbox)
            self.after(0, lambda: self._set_status("No users registered yet", "#ff9800"))
            return

        match = self.face_engine.find_best_match(face.embedding, users)

        if match:
            user, score = match
            self._last_result = ("match", (user, score, face.bbox))

            # cooldown check
            now = time.time()
            last_time = self._last_scan_times.get(user.id, 0)
            if now - last_time < SCAN_COOLDOWN_SECONDS:
                return  # skip, too soon

            self._last_scan_times[user.id] = now
            logged = self.db.log_attendance(user.id, score)

            # update UI on main thread
            self.after(0, lambda: self._show_match(user, score, logged))
        else:
            self._last_result = ("unknown", face.bbox)
            self.after(0, lambda: self._set_status("Face not recognized", "#e94560"))

    def _manual_scan(self):
        """Trigger a single scan right now."""
        frame = self.camera.get_frame()
        if frame is not None:
            threading.Thread(
                target=self._run_inference, args=(frame.copy(),), daemon=True
            ).start()

    def _show_match(self, user, score, newly_logged):
        self.status_icon.configure(text="✅" if newly_logged else "ℹ️")
        if newly_logged:
            self.status_text.configure(
                text="Attendance Recorded!", text_color="#4ecca3"
            )
        else:
            self.status_text.configure(
                text="Already checked in today", text_color="#ff9800"
            )

        self.name_label.configure(text=user.name)
        self.dept_label.configure(text=user.department)
        self.conf_label.configure(text=f"Confidence: {score:.1%}")

        from datetime import datetime
        self.time_label.configure(text=datetime.now().strftime("%I:%M %p"))

        self.detail_frame.pack(fill="x", padx=20, pady=10)
        self._update_stats()

    def _set_status(self, msg, color="#aaaaaa"):
        self.status_icon.configure(text="📷")
        self.status_text.configure(text=msg, text_color=color)
        self.detail_frame.pack_forget()

    def _toggle_auto_scan(self):
        self._auto_scan = self.auto_scan_switch.get()

    def _update_stats(self):
        reg = self.db.get_user_count()
        today = self.db.get_attendance_count()
        self.registered_label.configure(text=f"Registered: {reg}")
        self.today_label.configure(text=f"Today: {today} check-ins")

    def on_hide(self):
        """Called when navigating away from this screen."""
        self._running = False
