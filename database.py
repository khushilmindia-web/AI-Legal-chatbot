# from backend.database import *  # noqa: F401,F403
import json
import secrets
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import bcrypt


BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "db"
DB_PATH = DB_DIR / "app.db"
SESSION_DURATION_DAYS = 14


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat()


def session_expiry_iso() -> str:
    return (utc_now() + timedelta(days=SESSION_DURATION_DAYS)).isoformat()


def get_connection() -> sqlite3.Connection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


def column_exists(connection: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    columns = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(column["name"] == column_name for column in columns)


def init_db() -> None:
    with closing(get_connection()) as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                auth_provider TEXT NOT NULL DEFAULT 'local',
                google_sub TEXT UNIQUE,
                state TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        if not column_exists(connection, "users", "auth_provider"):
            cursor.execute("ALTER TABLE users ADD COLUMN auth_provider TEXT NOT NULL DEFAULT 'local'")
        if not column_exists(connection, "users", "google_sub"):
            cursor.execute("ALTER TABLE users ADD COLUMN google_sub TEXT")
        cursor.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_google_sub ON users (google_sub)"
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                source TEXT,
                confidence REAL,
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
            """
        )
        connection.commit()


def serialize_user(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    return {
        "id": row["id"],
        "full_name": row["full_name"],
        "email": row["email"],
        "state": row["state"],
        "created_at": row["created_at"],
    }


def get_user_by_email(email: str) -> dict | None:
    with closing(get_connection()) as connection:
        row = connection.execute(
            "SELECT * FROM users WHERE email = ?",
            (email.strip().lower(),),
        ).fetchone()
    return serialize_user(row)


def get_user_by_google_sub(google_sub: str) -> dict | None:
    with closing(get_connection()) as connection:
        row = connection.execute(
            "SELECT * FROM users WHERE google_sub = ?",
            (google_sub,),
        ).fetchone()
    return serialize_user(row)


def create_user(full_name: str, email: str, password: str, state: str | None = None) -> dict:
    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    normalized_email = email.strip().lower()
    with closing(get_connection()) as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO users (full_name, email, password_hash, auth_provider, state, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (full_name.strip(), normalized_email, password_hash, "local", state, utc_now_iso()),
        )
        user_id = cursor.lastrowid
        connection.commit()
        row = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    user = serialize_user(row)
    if not user:
        raise ValueError("Failed to create user")
    return user


def authenticate_user(email: str, password: str) -> dict | None:
    normalized_email = email.strip().lower()
    with closing(get_connection()) as connection:
        row = connection.execute(
            "SELECT * FROM users WHERE email = ?",
            (normalized_email,),
        ).fetchone()
    if row is None:
        return None
    if not bcrypt.checkpw(password.encode("utf-8"), row["password_hash"].encode("utf-8")):
        return None
    return serialize_user(row)


def create_google_user(full_name: str, email: str, google_sub: str, state: str | None = None) -> dict:
    normalized_email = email.strip().lower()
    placeholder_password = secrets.token_urlsafe(32)
    password_hash = bcrypt.hashpw(placeholder_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    with closing(get_connection()) as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO users (full_name, email, password_hash, auth_provider, google_sub, state, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (full_name.strip(), normalized_email, password_hash, "google", google_sub, state, utc_now_iso()),
        )
        user_id = cursor.lastrowid
        connection.commit()
        row = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    user = serialize_user(row)
    if not user:
        raise ValueError("Failed to create Google user")
    return user


def link_google_account(user_id: int, google_sub: str) -> dict | None:
    with closing(get_connection()) as connection:
        connection.execute(
            """
            UPDATE users
            SET google_sub = ?,
                auth_provider = CASE
                    WHEN auth_provider = 'local' THEN 'both'
                    ELSE auth_provider
                END
            WHERE id = ?
            """,
            (google_sub, user_id),
        )
        connection.commit()
        row = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return serialize_user(row)


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    with closing(get_connection()) as connection:
        connection.execute(
            """
            INSERT INTO sessions (user_id, token, created_at, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, token, utc_now_iso(), session_expiry_iso()),
        )
        connection.commit()
    return token


def get_user_by_token(token: str) -> dict | None:
    if not token:
        return None
    with closing(get_connection()) as connection:
        row = connection.execute(
            """
            SELECT users.*
            FROM sessions
            JOIN users ON users.id = sessions.user_id
            WHERE sessions.token = ?
            """,
            (token,),
        ).fetchone()
        if row is None:
            return None
        session_row = connection.execute(
            "SELECT expires_at FROM sessions WHERE token = ?",
            (token,),
        ).fetchone()
        if session_row is None:
            return None
        if datetime.fromisoformat(session_row["expires_at"]) < utc_now():
            connection.execute("DELETE FROM sessions WHERE token = ?", (token,))
            connection.commit()
            return None
    return serialize_user(row)


def delete_session(token: str) -> None:
    with closing(get_connection()) as connection:
        connection.execute("DELETE FROM sessions WHERE token = ?", (token,))
        connection.commit()


def store_chat_message(
    user_id: int,
    role: str,
    message: str,
    source: str | None = None,
    confidence: float | None = None,
    metadata: dict | None = None,
) -> None:
    with closing(get_connection()) as connection:
        connection.execute(
            """
            INSERT INTO chat_messages (user_id, role, message, source, confidence, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                role,
                message,
                source,
                confidence,
                json.dumps(metadata or {}),
                utc_now_iso(),
            ),
        )
        connection.commit()


def list_chat_messages(user_id: int, limit: int = 100) -> list[dict]:
    with closing(get_connection()) as connection:
        rows = connection.execute(
            """
            SELECT id, role, message, source, confidence, metadata_json, created_at
            FROM chat_messages
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    results = []
    for row in reversed(rows):
        metadata = {}
        if row["metadata_json"]:
            try:
                metadata = json.loads(row["metadata_json"])
            except json.JSONDecodeError:
                metadata = {}
        results.append(
            {
                "id": row["id"],
                "role": row["role"],
                "message": row["message"],
                "source": row["source"],
                "confidence": row["confidence"],
                "metadata": metadata,
                "created_at": row["created_at"],
            }
        )
    return results
