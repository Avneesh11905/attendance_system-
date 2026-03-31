"""
Attendance dashboard — view check-in records by date, export to CSV.
"""

from datetime import date, timedelta
import customtkinter as ctk
from tkinter import filedialog


class DashboardFrame(ctk.CTkFrame):
    def __init__(self, parent, database):
        super().__init__(parent, fg_color="transparent")

        self.db = database
        self._selected_date = date.today()

        self._build_ui()
        self._load_data()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # top bar — date controls + stats
        top = ctk.CTkFrame(self, corner_radius=12, fg_color="#1a1a2e")
        top.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        top.grid_columnconfigure(1, weight=1)

        # date navigation
        date_frame = ctk.CTkFrame(top, fg_color="transparent")
        date_frame.grid(row=0, column=0, padx=20, pady=15)

        ctk.CTkButton(
            date_frame, text="◀", width=35, height=35,
            command=self._prev_day,
            fg_color="#333333", hover_color="#555555",
        ).pack(side="left", padx=(0, 8))

        self.date_label = ctk.CTkLabel(
            date_frame, text="",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.date_label.pack(side="left", padx=8)

        ctk.CTkButton(
            date_frame, text="▶", width=35, height=35,
            command=self._next_day,
            fg_color="#333333", hover_color="#555555",
        ).pack(side="left", padx=(8, 0))

        ctk.CTkButton(
            date_frame, text="Today", width=70, height=35,
            command=self._go_today,
            fg_color="#16213e", hover_color="#1a3a5c",
        ).pack(side="left", padx=(15, 0))

        # stats cards
        stats_frame = ctk.CTkFrame(top, fg_color="transparent")
        stats_frame.grid(row=0, column=1, padx=20, pady=15, sticky="e")

        self.total_label = ctk.CTkLabel(
            stats_frame, text="Total: 0",
            font=ctk.CTkFont(size=13), text_color="#4ecca3",
        )
        self.total_label.pack(side="left", padx=15)

        self.registered_label = ctk.CTkLabel(
            stats_frame, text="Registered: 0",
            font=ctk.CTkFont(size=13), text_color="#aaaaaa",
        )
        self.registered_label.pack(side="left", padx=15)

        # export button
        ctk.CTkButton(
            stats_frame, text="📥 Export CSV",
            command=self._export_csv,
            fg_color="#e94560", hover_color="#c81e45",
            width=120, height=35,
        ).pack(side="left", padx=(15, 0))

        # table area
        table_container = ctk.CTkFrame(self, corner_radius=12, fg_color="#1a1a2e")
        table_container.grid(row=1, column=0, sticky="nsew")
        table_container.grid_columnconfigure(0, weight=1)
        table_container.grid_rowconfigure(1, weight=1)

        # table header
        header_frame = ctk.CTkFrame(table_container, fg_color="#16213e", corner_radius=0)
        header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))

        headers = ["#", "Name", "Department", "Check-in Time", "Confidence"]
        widths = [40, 200, 150, 150, 100]
        for i, (h, w) in enumerate(zip(headers, widths)):
            header_frame.grid_columnconfigure(i, weight=1 if i in (1, 2) else 0)
            lbl = ctk.CTkLabel(
                header_frame, text=h, width=w,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color="#e94560",
            )
            lbl.grid(row=0, column=i, padx=8, pady=8, sticky="w")

        # scrollable rows
        self.table_scroll = ctk.CTkScrollableFrame(
            table_container, fg_color="transparent",
        )
        self.table_scroll.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.table_scroll.grid_columnconfigure(0, weight=1)

    def _load_data(self):
        # update date label
        if self._selected_date == date.today():
            day_str = "Today"
        elif self._selected_date == date.today() - timedelta(days=1):
            day_str = "Yesterday"
        else:
            day_str = self._selected_date.strftime("%B %d, %Y")
        self.date_label.configure(text=day_str)

        # fetch
        records = self.db.get_attendance(self._selected_date)
        registered = self.db.get_user_count()

        self.total_label.configure(text=f"Present: {len(records)}")
        self.registered_label.configure(text=f"Registered: {registered}")

        # clear old rows
        for widget in self.table_scroll.winfo_children():
            widget.destroy()

        if not records:
            empty = ctk.CTkLabel(
                self.table_scroll, text="No attendance records for this date",
                font=ctk.CTkFont(size=14), text_color="#666666",
            )
            empty.pack(pady=40)
            return

        # populate rows
        for idx, record in enumerate(records):
            row_color = "#16213e" if idx % 2 == 0 else "transparent"
            row = ctk.CTkFrame(self.table_scroll, fg_color=row_color, corner_radius=4)
            row.pack(fill="x", pady=1)

            row.grid_columnconfigure(1, weight=1)
            row.grid_columnconfigure(2, weight=1)

            # format time
            try:
                from datetime import datetime
                ts = datetime.fromisoformat(record.timestamp)
                time_str = ts.strftime("%I:%M %p")
            except:
                time_str = record.timestamp

            values = [
                str(idx + 1),
                record.user_name,
                record.department,
                time_str,
                f"{record.confidence:.1%}",
            ]
            widths = [40, 200, 150, 150, 100]

            for col, (val, w) in enumerate(zip(values, widths)):
                lbl = ctk.CTkLabel(row, text=val, width=w, anchor="w")
                lbl.grid(row=0, column=col, padx=8, pady=6, sticky="w")

    def _prev_day(self):
        self._selected_date -= timedelta(days=1)
        self._load_data()

    def _next_day(self):
        if self._selected_date < date.today():
            self._selected_date += timedelta(days=1)
            self._load_data()

    def _go_today(self):
        self._selected_date = date.today()
        self._load_data()

    def _export_csv(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=f"attendance_{self._selected_date.isoformat()}.csv",
        )
        if filepath:
            self.db.export_csv(self._selected_date, filepath)

    def on_hide(self):
        pass
