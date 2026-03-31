"""
SQLite database for storing registered users (with face embeddings)
and attendance logs. DB file gets created at data/attendance.db automatically.
"""

import csv
import sqlite3
import numpy as np
from datetime import date
from typing import Optional
from dataclasses import dataclass

from app.config import DB_PATH
from app.utils.helpers import embedding_to_bytes, bytes_to_embedding


@dataclass
class User:
    id: int
    name: str
    employee_id: str
    department: str
    embedding: np.ndarray
    created_at: str


@dataclass
class AttendanceRecord:
    id: int
    user_name: str
    department: str
    timestamp: str
    confidence: float


class Database:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._get_conn()
        c = conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                employee_id TEXT UNIQUE NOT NULL,
                department TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS attendance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                confidence REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)

        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_attendance_date
            ON attendance_logs(date(timestamp))
        """)

        conn.commit()
        conn.close()

    # --- User management ---

    def add_user(self, name, employee_id, department, embedding):
        """Register a new user. Returns False if employee_id is taken."""
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT INTO users (name, employee_id, department, embedding) VALUES (?, ?, ?, ?)",
                (name, employee_id, department, embedding_to_bytes(embedding)),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()

    def get_all_users(self) -> list[User]:
        """Load all users with embeddings (used for matching during scan)."""
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM users ORDER BY name").fetchall()
        conn.close()
        return [
            User(
                id=r["id"], name=r["name"], employee_id=r["employee_id"],
                department=r["department"],
                embedding=bytes_to_embedding(r["embedding"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def get_user_count(self):
        conn = self._get_conn()
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        conn.close()
        return count

    def delete_user(self, user_id):
        conn = self._get_conn()
        cur = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        deleted = cur.rowcount > 0
        conn.close()
        return deleted

    # --- Attendance ---

    def log_attendance(self, user_id, confidence):
        """Log check-in. Returns False if already checked in today."""
        conn = self._get_conn()
        today = date.today().isoformat()

        already = conn.execute(
            "SELECT id FROM attendance_logs WHERE user_id = ? AND date(timestamp) = ?",
            (user_id, today),
        ).fetchone()

        if already:
            conn.close()
            return False

        conn.execute(
            "INSERT INTO attendance_logs (user_id, confidence) VALUES (?, ?)",
            (user_id, confidence),
        )
        conn.commit()
        conn.close()
        return True

    def get_attendance(self, target_date=None) -> list[AttendanceRecord]:
        """Fetch attendance for a date (defaults to today)."""
        if target_date is None:
            target_date = date.today()

        conn = self._get_conn()
        rows = conn.execute("""
            SELECT a.id, u.name as user_name, u.department, a.timestamp, a.confidence
            FROM attendance_logs a
            JOIN users u ON a.user_id = u.id
            WHERE date(a.timestamp) = ?
            ORDER BY a.timestamp
        """, (target_date.isoformat(),)).fetchall()
        conn.close()

        return [
            AttendanceRecord(
                id=r["id"], user_name=r["user_name"], department=r["department"],
                timestamp=r["timestamp"], confidence=r["confidence"],
            )
            for r in rows
        ]

    def get_attendance_count(self, target_date=None):
        if target_date is None:
            target_date = date.today()
        conn = self._get_conn()
        count = conn.execute(
            "SELECT COUNT(*) FROM attendance_logs WHERE date(timestamp) = ?",
            (target_date.isoformat(),),
        ).fetchone()[0]
        conn.close()
        return count

    def export_csv(self, target_date, filepath):
        """Dump attendance for a given date into a CSV file."""
        records = self.get_attendance(target_date)
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Department", "Check-in Time", "Confidence"])
            for r in records:
                writer.writerow([r.user_name, r.department, r.timestamp, f"{r.confidence:.2f}"])
