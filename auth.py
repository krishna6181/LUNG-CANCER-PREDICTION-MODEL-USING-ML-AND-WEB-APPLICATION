"""User authentication and database management."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash


DB_PATH = Path(__file__).resolve().parents[1] / "users.db"


def init_db():
    """Initialize the user database if it doesn't exist."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


def create_user(username: str, email: str, password: str) -> bool:
    """Create a new user. Returns True if successful, False if username/email already exists."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            password_hash = generate_password_hash(password)
            cursor.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username, email, password_hash),
            )
            conn.commit()
            return True
    except sqlite3.IntegrityError:
        return False
    except sqlite3.OperationalError:
        # Database is locked or other operational error
        return False


def verify_user(username: str, password: str) -> dict | None:
    """Verify user credentials. Returns user dict if valid, None otherwise."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, username, email, password_hash FROM users WHERE username = ?",
                (username,),
            )
            user = cursor.fetchone()

            if user and check_password_hash(user[3], password):
                return {
                    "id": user[0],
                    "username": user[1],
                    "email": user[2],
                }
            return None
    except sqlite3.Error:
        return None


def get_user_by_id(user_id: int) -> dict | None:
    """Get user by ID. Returns user dict if found, None otherwise."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, username, email FROM users WHERE id = ?",
                (user_id,),
            )
            user = cursor.fetchone()

            if user:
                return {
                    "id": user[0],
                    "username": user[1],
                    "email": user[2],
                }
            return None
    except sqlite3.Error:
        return None


# Initialize database on import
init_db()

