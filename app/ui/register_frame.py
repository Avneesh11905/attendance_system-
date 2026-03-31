"""
Registration screen — capture a face and register a new user.
"""

import threading
import customtkinter as ctk
from app.config import CAMERA_DISPLAY_WIDTH, CAMERA_DISPLAY_HEIGHT
from app.utils.helpers import frame_to_ctkimage, draw_face_box


class RegisterFrame(ctk.CTkFrame):
    def __init__(self, parent, face_engine, camera, database):
        super().__init__(parent, fg_color="transparent")

        self.face_engine = face_engine
        self.camera = camera
        self.db = database

        self._running = True
        self._captured_frame = None
        self._captured_embedding = None

        self._build_ui()

        if not self.camera.is_running:
            try:
                self.camera.start()
            except RuntimeError as e:
                self._show_msg(str(e), "red")

        self._update_feed()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        # left — camera
        cam_frame = ctk.CTkFrame(self, corner_radius=12, fg_color="#1a1a2e")
        cam_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        header = ctk.CTkLabel(
            cam_frame, text="Register New User",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        header.pack(pady=(15, 10))

        self.camera_label = ctk.CTkLabel(cam_frame, text="Starting camera...")
        self.camera_label.pack(padx=15, pady=(0, 10))

        btn_row = ctk.CTkFrame(cam_frame, fg_color="transparent")
        btn_row.pack(pady=(0, 15))

        self.capture_btn = ctk.CTkButton(
            btn_row, text="📸 Capture Face",
            command=self._capture_face,
            fg_color="#e94560", hover_color="#c81e45",
            width=140,
        )
        self.capture_btn.pack(side="left", padx=5)

        self.retake_btn = ctk.CTkButton(
            btn_row, text="🔄 Retake",
            command=self._retake,
            fg_color="#333333", hover_color="#555555",
            width=100, state="disabled",
        )
        self.retake_btn.pack(side="left", padx=5)

        # right — form
        form_frame = ctk.CTkFrame(self, corner_radius=12, fg_color="#1a1a2e")
        form_frame.grid(row=0, column=1, sticky="nsew")

        form_header = ctk.CTkLabel(
            form_frame, text="User Details",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        form_header.pack(pady=(15, 20))

        # name
        ctk.CTkLabel(form_frame, text="Full Name", text_color="#aaaaaa").pack(
            padx=25, anchor="w"
        )
        self.name_entry = ctk.CTkEntry(
            form_frame, placeholder_text="e.g. John Doe", height=38
        )
        self.name_entry.pack(fill="x", padx=25, pady=(3, 12))

        # employee id
        ctk.CTkLabel(form_frame, text="Employee / Student ID", text_color="#aaaaaa").pack(
            padx=25, anchor="w"
        )
        self.id_entry = ctk.CTkEntry(
            form_frame, placeholder_text="e.g. EMP-001", height=38
        )
        self.id_entry.pack(fill="x", padx=25, pady=(3, 12))

        # department
        ctk.CTkLabel(form_frame, text="Department", text_color="#aaaaaa").pack(
            padx=25, anchor="w"
        )
        self.dept_entry = ctk.CTkEntry(
            form_frame, placeholder_text="e.g. Engineering", height=38
        )
        self.dept_entry.pack(fill="x", padx=25, pady=(3, 20))

        # register button
        self.register_btn = ctk.CTkButton(
            form_frame, text="✓ Register User",
            command=self._register_user,
            fg_color="#4ecca3", hover_color="#3ba882",
            text_color="#000000",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=42,
            state="disabled",
        )
        self.register_btn.pack(fill="x", padx=25, pady=(5, 10))

        # status message
        self.msg_label = ctk.CTkLabel(
            form_frame, text="", font=ctk.CTkFont(size=12),
            wraplength=250,
        )
        self.msg_label.pack(pady=(5, 15))

        # face preview
        self.preview_label = ctk.CTkLabel(
            form_frame, text="No face captured yet",
            font=ctk.CTkFont(size=11), text_color="#666666",
        )
        self.preview_label.pack(pady=(5, 15))

    def _update_feed(self):
        if not self._running:
            return

        # don't update camera when we have a frozen capture
        if self._captured_frame is None:
            frame = self.camera.get_frame()
            if frame is not None:
                img = frame_to_ctkimage(frame, CAMERA_DISPLAY_WIDTH, CAMERA_DISPLAY_HEIGHT)
                self.camera_label.configure(image=img, text="")
                self.camera_label._image = img

        self.after(33, self._update_feed)

    def _capture_face(self):
        """Freeze current frame and try to detect a face."""
        frame = self.camera.get_frame()
        if frame is None:
            self._show_msg("No camera frame available", "#e94560")
            return

        self._show_msg("Detecting face...", "#aaaaaa")

        def do_detect():
            faces = self.face_engine.detect_and_embed(frame)
            if not faces:
                self.after(0, lambda: self._show_msg("No face found — try again", "#e94560"))
                return

            face = faces[0]
            self._captured_embedding = face.embedding

            # draw box on the frozen frame
            display = frame.copy()
            draw_face_box(display, face.bbox, "Detected", color=(0, 255, 0))

            self._captured_frame = display
            img = frame_to_ctkimage(display, CAMERA_DISPLAY_WIDTH, CAMERA_DISPLAY_HEIGHT)

            def update_ui():
                self.camera_label.configure(image=img, text="")
                self.camera_label._image = img
                self.capture_btn.configure(state="disabled")
                self.retake_btn.configure(state="normal")
                self.register_btn.configure(state="normal")
                self.preview_label.configure(text="✓ Face captured", text_color="#4ecca3")
                self._show_msg("Face captured! Fill in the details and click Register.", "#4ecca3")

            self.after(0, update_ui)

        threading.Thread(target=do_detect, daemon=True).start()

    def _retake(self):
        """Go back to live camera."""
        self._captured_frame = None
        self._captured_embedding = None
        self.capture_btn.configure(state="normal")
        self.retake_btn.configure(state="disabled")
        self.register_btn.configure(state="disabled")
        self.preview_label.configure(text="No face captured yet", text_color="#666666")
        self._show_msg("", "#aaaaaa")

    def _register_user(self):
        name = self.name_entry.get().strip()
        emp_id = self.id_entry.get().strip()
        dept = self.dept_entry.get().strip()

        if not name or not emp_id or not dept:
            self._show_msg("Please fill in all fields", "#e94560")
            return

        if self._captured_embedding is None:
            self._show_msg("Capture a face first", "#e94560")
            return

        success = self.db.add_user(name, emp_id, dept, self._captured_embedding)

        if success:
            self._show_msg(f"Registered {name} successfully!", "#4ecca3")
            # clear form
            self.name_entry.delete(0, "end")
            self.id_entry.delete(0, "end")
            self.dept_entry.delete(0, "end")
            self._retake()
        else:
            self._show_msg(f"Employee ID '{emp_id}' is already registered", "#e94560")

    def _show_msg(self, text, color="#aaaaaa"):
        self.msg_label.configure(text=text, text_color=color)

    def on_hide(self):
        self._running = False
