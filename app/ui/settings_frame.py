"""
Settings screen — adjust threshold, pick camera, toggle options.
"""

import customtkinter as ctk
from app.config import SIMILARITY_THRESHOLD, DEFAULT_CAMERA_INDEX


class SettingsFrame(ctk.CTkFrame):
    def __init__(self, parent, camera, face_engine):
        super().__init__(parent, fg_color="transparent")

        self.camera = camera
        self.face_engine = face_engine

        self._build_ui()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)

        # title
        title = ctk.CTkLabel(
            self, text="Settings",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        title.grid(row=0, column=0, pady=(0, 20), sticky="w")

        # --- recognition settings ---
        rec_card = ctk.CTkFrame(self, corner_radius=12, fg_color="#1a1a2e")
        rec_card.grid(row=1, column=0, sticky="ew", pady=(0, 15))

        ctk.CTkLabel(
            rec_card, text="Recognition",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).pack(anchor="w", padx=25, pady=(15, 10))

        # threshold slider
        thresh_row = ctk.CTkFrame(rec_card, fg_color="transparent")
        thresh_row.pack(fill="x", padx=25, pady=(0, 5))

        ctk.CTkLabel(
            thresh_row, text="Similarity Threshold",
            text_color="#aaaaaa",
        ).pack(side="left")

        self.thresh_value = ctk.CTkLabel(
            thresh_row, text=f"{SIMILARITY_THRESHOLD:.0%}",
            text_color="#4ecca3", font=ctk.CTkFont(weight="bold"),
        )
        self.thresh_value.pack(side="right")

        self.thresh_slider = ctk.CTkSlider(
            rec_card, from_=0.2, to=0.8,
            number_of_steps=12,
            command=self._on_threshold_change,
        )
        self.thresh_slider.set(SIMILARITY_THRESHOLD)
        self.thresh_slider.pack(fill="x", padx=25, pady=(0, 5))

        ctk.CTkLabel(
            rec_card,
            text="Lower = more lenient (may cause false matches)\n"
                 "Higher = stricter (may reject valid faces)",
            font=ctk.CTkFont(size=11), text_color="#666666",
            justify="left",
        ).pack(anchor="w", padx=25, pady=(0, 15))

        # --- camera settings ---
        cam_card = ctk.CTkFrame(self, corner_radius=12, fg_color="#1a1a2e")
        cam_card.grid(row=2, column=0, sticky="ew", pady=(0, 15))

        ctk.CTkLabel(
            cam_card, text="Camera",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).pack(anchor="w", padx=25, pady=(15, 10))

        cam_row = ctk.CTkFrame(cam_card, fg_color="transparent")
        cam_row.pack(fill="x", padx=25, pady=(0, 5))

        ctk.CTkLabel(cam_row, text="Camera Device", text_color="#aaaaaa").pack(
            side="left"
        )

        # detect available cameras
        from app.core.camera import CameraManager
        cams = CameraManager.list_cameras()
        cam_options = [f"Camera {i}" for i in cams] if cams else ["No camera found"]

        self.cam_dropdown = ctk.CTkOptionMenu(
            cam_row,
            values=cam_options,
            command=self._on_camera_change,
            fg_color="#16213e",
            button_color="#333333",
            button_hover_color="#555555",
            width=150,
        )
        current = f"Camera {self.camera.camera_index}"
        if current in cam_options:
            self.cam_dropdown.set(current)
        self.cam_dropdown.pack(side="right")

        ctk.CTkLabel(
            cam_card,
            text="Switch between connected cameras (webcam, USB, etc.)",
            font=ctk.CTkFont(size=11), text_color="#666666",
        ).pack(anchor="w", padx=25, pady=(5, 15))

        # --- about ---
        about_card = ctk.CTkFrame(self, corner_radius=12, fg_color="#1a1a2e")
        about_card.grid(row=3, column=0, sticky="ew", pady=(0, 15))

        ctk.CTkLabel(
            about_card, text="About",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).pack(anchor="w", padx=25, pady=(15, 10))

        info_lines = [
            ("Model", "InsightFace buffalo_l (SCRFD + ArcFace)"),
            ("Runtime", "ONNX Runtime (CPU)"),
            ("Embedding", "512-dimensional face vector"),
            ("Database", "SQLite (local)"),
        ]

        for label, value in info_lines:
            row = ctk.CTkFrame(about_card, fg_color="transparent")
            row.pack(fill="x", padx=25, pady=2)
            ctk.CTkLabel(row, text=label, text_color="#aaaaaa", width=100, anchor="w").pack(side="left")
            ctk.CTkLabel(row, text=value, text_color="#ffffff").pack(side="left")

        ctk.CTkLabel(
            about_card, text="",  # spacer
        ).pack(pady=5)

    def _on_threshold_change(self, value):
        # update the config at runtime
        import app.config as cfg
        cfg.SIMILARITY_THRESHOLD = value
        self.thresh_value.configure(text=f"{value:.0%}")

    def _on_camera_change(self, choice):
        try:
            idx = int(choice.split(" ")[1])
            self.camera.switch_camera(idx)
        except (ValueError, IndexError):
            pass

    def on_hide(self):
        pass
