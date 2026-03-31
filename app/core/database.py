"""
Database Layer — SQLite operations for user registration and attendance logging.

Handles:
- User table: stores name, employee ID, department, and face embedding (as BLOB)
- Attendance log table: timestamps each check-in with user reference and confidence score
- Duplicate prevention: one check-in per user per day
- CSV export for attendance reports

The database file is created automatically at data/attendance.db on first run.
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
    """Registered user with their face embedding."""
    id: int
    name: str
    employee_id: str
    department: str
    embedding: np.ndarray
    created_at: str


@dataclass
class AttendanceRecord:
    """A single attendance log entry (joined with user data)."""
    id: int
    user_name: str
    department: str
    timestamp: str
    confidence: float


class Database:
    """
    SQLite database manager for the attendance system.

    Creates and manages two tables:
    - users: registered faces with 512-D embeddings stored as BLOBs
    - attendance_logs: timestamped check-in records

    All methods use short-lived connections (no connection pooling needed
    for a single-user desktop app).
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Create a new connection with Row factory for dict-like access."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                employee_id TEXT UNIQUE NOT NULL,
                department TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                confidence REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)

        # Index for fast date-based attendance queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_attendance_date
            ON attendance_logs(date(timestamp))
        """)

        conn.commit()
        conn.close()
        print(f"[Database] Initialized at {self.db_path}")

    # ──────────────────────────────────────────────
    # User Management
    # ──────────────────────────────────────────────

    def add_user(self, name: str, employee_id: str, department: str, embedding: np.ndarray) -> bool:
        """
        Register a new user with their face embedding.

        Args:
            name: Full name
            employee_id: Unique employee/student ID
            department: Department or class
            embedding: 512-D numpy array from ArcFace

        Returns:
            True if registered successfully, False if employee_id already exists.
        """
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT INTO users (name, employee_id, department, embedding) VALUES (?, ?, ?, ?)",
                (name, employee_id, department, embedding_to_bytes(embedding)),
            )
            conn.commit()
            print(f"[Database] Registered user: {name} ({employee_id})")
            return True
        except sqlite3.IntegrityError:
            print(f"[Database] Duplicate employee_id: {employee_id}")
            return False
        finally:
            conn.close()

    def get_all_users(self) -> list[User]:
        """
        Retrieve all registered users with their embeddings.
        Used during recognition to compare against scanned faces.
        """
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM users ORDER BY name").fetchall()
        conn.close()

        return [
            User(
                id=row["id"],
                name=row["name"],
                employee_id=row["employee_id"],
                department=row["department"],
                embedding=bytes_to_embedding(row["embedding"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def get_user_count(self) -> int:
        """Get total number of registered users."""
        conn = self._get_conn()
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        conn.close()
        return count

    def delete_user(self, user_id: int) -> bool:
        """Remove a registered user by ID."""
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        deleted = cursor.rowcount > 0
        conn.close()
        return deleted

    # ──────────────────────────────────────────────
    # Attendance Logging
    # ──────────────────────────────────────────────

    def log_attendance(self, user_id: int, confidence: float) -> bool:
        """
        Log a check-in for a user.

        Prevents duplicate entries: each user can only check in once per day.

        Args:
            user_id: The matched user's database ID
            confidence: Cosine similarity score from the match

        Returns:
            True if attendance was logged, False if user already checked in today.
        """
        conn = self._get_conn()
        today = date.today().isoformat()

        # Check for existing check-in today
        existing = conn.execute(
            "SELECT id FROM attendance_logs WHERE user_id = ? AND date(timestamp) = ?",
            (user_id, today),
        ).fetchone()

        if existing:
            conn.close()
            return False

        conn.execute(
            "INSERT INTO attendance_logs (user_id, confidence) VALUES (?, ?)",
            (user_id, confidence),
        )
        conn.commit()
        conn.close()
        print(f"[Database] Attendance logged for user_id={user_id}, confidence={confidence:.3f}")
        return True

    def get_attendance(self, target_date: Optional[date] = None) -> list[AttendanceRecord]:
        """
        Get attendance records for a specific date.

        Args:
            target_date: The date to query. Defaults to today.

        Returns:
            List of AttendanceRecord objects with user info joined in.
        """
        if target_date is None:
            target_date = date.today()

        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT a.id, u.name AS user_name, u.department, a.timestamp, a.confidence
            FROM attendance_logs a
            JOIN users u ON a.user_id = u.id
            WHERE date(a.timestamp) = ?
            ORDER BY a.timestamp
            """,
            (target_date.isoformat(),),
        ).fetchall()
        conn.close()

        return [
            AttendanceRecord(
                id=row["id"],
                user_name=row["user_name"],
                department=row["department"],
                timestamp=row["timestamp"],
                confidence=row["confidence"],
            )
            for row in rows
        ]

    def get_attendance_count(self, target_date: Optional[date] = None) -> int:
        """Get the number of check-ins for a date."""
        if target_date is None:
            target_date = date.today()

        conn = self._get_conn()
        count = conn.execute(
            "SELECT COUNT(*) FROM attendance_logs WHERE date(timestamp) = ?",
            (target_date.isoformat(),),
        ).fetchone()[0]
        conn.close()
        return count

    # ──────────────────────────────────────────────
    # Export
    # ──────────────────────────────────────────────

    def export_csv(self, target_date: date, filepath: str):
        """
        Export attendance records for a date to a CSV file.

        Args:
            target_date: Date to export
            filepath: Full path for the output CSV file
        """
        records = self.get_attendance(target_date)
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Department", "Check-in Time", "Confidence"])
            for r in records:
                writer.writerow([r.user_name, r.department, r.timestamp, f"{r.confidence:.2f}"])
        print(f"[Database] Exported {len(records)} records to {filepath}")
