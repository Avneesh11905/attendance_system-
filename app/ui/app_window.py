"""
Main application window with sidebar navigation.
Swaps between different screens (scan, register, dashboard, settings).
"""

import customtkinter as ctk
from app.config import APP_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT


class AppWindow(ctk.CTk):
    def __init__(self, face_engine, camera, database):
        super().__init__()

        self.face_engine = face_engine
        self.camera = camera
        self.database = database

        # window setup
        self.title(APP_TITLE)
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.minsize(900, 600)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # layout: sidebar on left, content on right
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_content_area()

        # load the scan screen by default
        self._current_frame = None
        self.show_frame("scan")

        # clean up camera on close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_sidebar(self):
        sidebar = ctk.CTkFrame(self, width=200, corner_radius=0, fg_color="#1a1a2e")
        sidebar.grid(row=0, column=0, sticky="nsw")
        sidebar.grid_propagate(False)

        # app title
        title_label = ctk.CTkLabel(
            sidebar, text="FaceAttend",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#e94560",
        )
        title_label.pack(pady=(30, 5))

        subtitle = ctk.CTkLabel(
            sidebar, text="Biometric Attendance",
            font=ctk.CTkFont(size=11),
            text_color="#888888",
        )
        subtitle.pack(pady=(0, 30))

        # nav buttons
        nav_items = [
            ("📷  Scan", "scan"),
            ("➕  Register", "register"),
            ("📊  Dashboard", "dashboard"),
            ("⚙️  Settings", "settings"),
        ]

        self._nav_buttons = {}
        for label_text, frame_name in nav_items:
            btn = ctk.CTkButton(
                sidebar, text=label_text,
                font=ctk.CTkFont(size=14),
                fg_color="transparent",
                text_color="#ffffff",
                hover_color="#16213e",
                anchor="w",
                height=40,
                corner_radius=8,
                command=lambda name=frame_name: self.show_frame(name),
            )
            btn.pack(fill="x", padx=15, pady=3)
            self._nav_buttons[frame_name] = btn

        # bottom spacer + version
        sidebar.pack_propagate(False)
        spacer = ctk.CTkLabel(sidebar, text="")
        spacer.pack(expand=True)

        version = ctk.CTkLabel(
            sidebar, text="v1.0.0",
            font=ctk.CTkFont(size=10),
            text_color="#555555",
        )
        version.pack(pady=(0, 15))

    def _build_content_area(self):
        self.content_area = ctk.CTkFrame(self, fg_color="transparent")
        self.content_area.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.content_area.grid_columnconfigure(0, weight=1)
        self.content_area.grid_rowconfigure(0, weight=1)

    def show_frame(self, name):
        """Switch to a different screen."""
        # stop any updates in the current frame
        if self._current_frame is not None:
            if hasattr(self._current_frame, "on_hide"):
                self._current_frame.on_hide()
            self._current_frame.destroy()

        # highlight active nav button
        for btn_name, btn in self._nav_buttons.items():
            if btn_name == name:
                btn.configure(fg_color="#16213e")
            else:
                btn.configure(fg_color="transparent")

        # import here to avoid circular imports
        if name == "scan":
            from app.ui.scan_frame import ScanFrame
            self._current_frame = ScanFrame(
                self.content_area, self.face_engine, self.camera, self.database
            )
        elif name == "register":
            from app.ui.register_frame import RegisterFrame
            self._current_frame = RegisterFrame(
                self.content_area, self.face_engine, self.camera, self.database
            )
        elif name == "dashboard":
            from app.ui.dashboard_frame import DashboardFrame
            self._current_frame = DashboardFrame(self.content_area, self.database)
        elif name == "settings":
            from app.ui.settings_frame import SettingsFrame
            self._current_frame = SettingsFrame(
                self.content_area, self.camera, self.face_engine
            )

        self._current_frame.grid(row=0, column=0, sticky="nsew")

    def _on_close(self):
        """Clean up before closing."""
        if self._current_frame and hasattr(self._current_frame, "on_hide"):
            self._current_frame.on_hide()
        self.camera.stop()
        self.destroy()
