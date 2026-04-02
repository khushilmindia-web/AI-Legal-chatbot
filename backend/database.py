import json
import hashlib
import json
import logging
import secrets
import sqlite3

from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import bcrypt

from .paths import APP_DB_PATH, VECTOR_DB_DIR

DB_DIR = VECTOR_DB_DIR
DB_PATH = APP_DB_PATH

SESSION_DURATION_DAYS = 14
PASSWORD_RESET_DURATION_MINUTES = 30
logger = logging.getLogger("chatbot.database")

def safe_log_value(value: str | None) -> str:
    if not value:
        return "none"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def utc_now_iso() -> str:
    return utc_now().isoformat()

def coerce_iso_timestamp(value: str | None, fallback: str | None = None) -> str:
    candidate = (value or fallback or "").strip()
    if candidate:
        try:
            datetime.fromisoformat(candidate)
            return candidate
        except ValueError:
            pass
    return fallback or utc_now_iso()

def session_expiry_iso() -> str:
    return (utc_now() + timedelta(days=SESSION_DURATION_DAYS)).isoformat()

def password_reset_expiry_iso() -> str:
    return (utc_now() + timedelta(minutes=PASSWORD_RESET_DURATION_MINUTES)).isoformat()

def get_connection() -> sqlite3.Connection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    connection.row_factory = sqlite3.Row

    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
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
                chat_id INTEGER,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                source TEXT,
                confidence REAL,
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                FOREIGN KEY (chat_id) REFERENCES chat_sessions (id) ON DELETE CASCADE
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT,
                created_at TEXT NOT NULL,
                last_activity TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
            """
        )

        if not column_exists(connection, "chat_messages", "chat_id"):
            cursor.execute("ALTER TABLE chat_messages ADD COLUMN chat_id INTEGER")

            existing_users = cursor.execute(
                "SELECT DISTINCT user_id FROM chat_messages WHERE chat_id IS NULL"
            ).fetchall()
            for row in existing_users:
                user_id = row["user_id"]
                now = utc_now_iso()
                cursor.execute(
                    "INSERT INTO chat_sessions (user_id, title, created_at, last_activity) VALUES (?, ?, ?, ?)",
                    (user_id, "Default Chat", now, now),
                )
                default_chat_id = cursor.lastrowid
                cursor.execute(
                    "UPDATE chat_messages SET chat_id = ? WHERE user_id = ? AND chat_id IS NULL",
                    (default_chat_id, user_id),
                )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                used_at TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS matter_intakes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                chat_id INTEGER,
                status TEXT NOT NULL DEFAULT 'active',
                initial_query TEXT,
                issue_summary TEXT,
                state TEXT,
                district TEXT,
                matter_type TEXT,
                matter_stage TEXT,
                is_own_matter INTEGER,
                urgency TEXT,
                last_question_key TEXT,
                relief_goal TEXT,
                details_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                FOREIGN KEY (chat_id) REFERENCES chat_sessions (id) ON DELETE CASCADE
            )
            """
        )

        if not column_exists(connection, "matter_intakes", "chat_id"):
            cursor.execute("ALTER TABLE matter_intakes ADD COLUMN chat_id INTEGER")

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_matter_intakes_user_status ON matter_intakes (user_id, status)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_matter_intakes_user_chat_status ON matter_intakes (user_id, chat_id, status)"
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS uploaded_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                matter_intake_id INTEGER,
                original_name TEXT NOT NULL,
                stored_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                content_type TEXT,
                file_size INTEGER,
                document_kind TEXT,
                description TEXT,
                extracted_text TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                FOREIGN KEY (matter_intake_id) REFERENCES matter_intakes (id) ON DELETE SET NULL
            )
            """
        )

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_uploaded_documents_user_matter ON uploaded_documents (user_id, matter_intake_id)"
        )

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_password_reset_tokens_user_id ON password_reset_tokens (user_id)"
        )

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions (expires_at)"
        )

        connection.commit()

def check_db_health() -> dict:
    try:
        with closing(get_connection()) as connection:
            connection.execute("SELECT 1").fetchone()
            connection.execute("BEGIN IMMEDIATE")
            connection.rollback()
        return {"ok": True, "error": None}

    except sqlite3.OperationalError as exc:
        logger.error("database health check failed error=%s", exc)
        return {"ok": False, "error": str(exc)}

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
    normalized_email = email.strip().lower()

    try:
        with closing(get_connection()) as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE email = ?",
                (normalized_email,),
            ).fetchone()

    except sqlite3.OperationalError as exc:
        logger.error("get_user_by_email failed for email_hash=%s error=%s", safe_log_value(normalized_email), exc)
        raise
    return serialize_user(row)

def get_user_by_google_sub(google_sub: str) -> dict | None:
    try:
        with closing(get_connection()) as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE google_sub = ?",
                (google_sub,),
            ).fetchone()

    except sqlite3.OperationalError as exc:
        logger.error("get_user_by_google_sub failed for sub_hash=%s error=%s", safe_log_value(google_sub), exc)
        raise
    return serialize_user(row)

def create_user(full_name: str, email: str, password: str, state: str | None = None) -> dict:
    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    normalized_email = email.strip().lower()

    try:
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

    except sqlite3.OperationalError as exc:
        logger.error("create_user failed for email_hash=%s error=%s", safe_log_value(normalized_email), exc)
        raise
    user = serialize_user(row)

    if not user:
        raise ValueError("Failed to create user")
    return user

def authenticate_user(email: str, password: str) -> dict | None:
    normalized_email = email.strip().lower()

    try:
        with closing(get_connection()) as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE email = ?",
                (normalized_email,),
            ).fetchone()

    except sqlite3.OperationalError as exc:
        logger.error("authenticate_user lookup failed for email_hash=%s error=%s", safe_log_value(normalized_email), exc)
        raise
    
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

def cleanup_expired_sessions() -> int:
    with closing(get_connection()) as connection:
        cursor = connection.execute(
            "DELETE FROM sessions WHERE expires_at <= ?",
            (utc_now_iso(),),
        )
        connection.commit()
    return cursor.rowcount

def create_session(user_id: int) -> str:
    cleanup_expired_sessions()
    token = secrets.token_urlsafe(32)

    try:
        with closing(get_connection()) as connection:
            connection.execute(
                """
                INSERT INTO sessions (user_id, token, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, token, utc_now_iso(), session_expiry_iso()),
            )
            connection.commit()

    except sqlite3.OperationalError as exc:
        logger.error("create_session failed for user_id=%s error=%s", user_id, exc)
        raise
    return token

def get_user_by_token(token: str) -> dict | None:

    if not token:
        return None

    try:
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

    except sqlite3.OperationalError as exc:
        logger.error("get_user_by_token failed error=%s", exc)
        raise
    return serialize_user(row)

def delete_session(token: str) -> None:
    with closing(get_connection()) as connection:
        connection.execute("DELETE FROM sessions WHERE token = ?", (token,))
        connection.commit()

def cleanup_expired_password_reset_tokens() -> int:
    with closing(get_connection()) as connection:
        cursor = connection.execute(
            "DELETE FROM password_reset_tokens WHERE used_at IS NOT NULL OR expires_at <= ?",
            (utc_now_iso(),),
        )
        connection.commit()
    return cursor.rowcount

def create_password_reset_token(user_id: int) -> str:
    cleanup_expired_password_reset_tokens()
    token = secrets.token_urlsafe(32)
    with closing(get_connection()) as connection:
        connection.execute(
            "DELETE FROM password_reset_tokens WHERE user_id = ?",
            (user_id,),
        )

        connection.execute(
            """
            INSERT INTO password_reset_tokens (user_id, token, created_at, expires_at, used_at)
            VALUES (?, ?, ?, ?, NULL)
            """,
            (user_id, token, utc_now_iso(), password_reset_expiry_iso()),
        )

        connection.commit()
    return token

def consume_password_reset_token(token: str, new_password: str) -> dict | None:
    normalized_token = (token or "").strip()
    if not normalized_token:
        return None

    cleanup_expired_password_reset_tokens()
    with closing(get_connection()) as connection:
        row = connection.execute(
            """
            SELECT user_id, expires_at, used_at
            FROM password_reset_tokens
            WHERE token = ?
            """,
            (normalized_token,),
        ).fetchone()
        if row is None:
            return None
        
        if row["used_at"] is not None:
            return None
        
        if datetime.fromisoformat(row["expires_at"]) < utc_now():
            connection.execute("DELETE FROM password_reset_tokens WHERE token = ?", (normalized_token,))
            connection.commit()
            return None

        password_hash = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        connection.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (password_hash, row["user_id"]),
        )

        connection.execute(
            "UPDATE password_reset_tokens SET used_at = ? WHERE token = ?",
            (utc_now_iso(), normalized_token),
        )

        connection.execute(
            "DELETE FROM sessions WHERE user_id = ?",
            (row["user_id"],),
        )
        connection.commit()

        user_row = connection.execute(
            "SELECT * FROM users WHERE id = ?",
            (row["user_id"],),
        ).fetchone()
    return serialize_user(user_row)

def create_chat_session(user_id: int, title: str | None = None) -> dict:
    now = utc_now_iso()
    with closing(get_connection()) as connection:
        cursor = connection.execute(
            """
            INSERT INTO chat_sessions (user_id, title, created_at, last_activity)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, title or "New Chat", now, now),
        )
        connection.commit()
        session_id = cursor.lastrowid
        return {
            "id": session_id,
            "user_id": user_id,
            "title": title or "New Chat",
            "created_at": now,
            "last_activity": now,
        }


def update_chat_session_title(chat_id: int, title: str | None) -> None:
    cleaned_title = (title or "").strip()[:80]
    if not cleaned_title:
        return
    with closing(get_connection()) as connection:
        connection.execute(
            "UPDATE chat_sessions SET title = ? WHERE id = ?",
            (cleaned_title, chat_id),
        )
        connection.commit()


def get_chat_session(chat_id: int) -> dict | None:
    with closing(get_connection()) as connection:
        row = connection.execute(
            "SELECT id, user_id, title, created_at, last_activity FROM chat_sessions WHERE id = ?",
            (chat_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "title": row["title"],
            "created_at": coerce_iso_timestamp(row["created_at"]),
            "last_activity": coerce_iso_timestamp(row["last_activity"], row["created_at"]),
        }


def list_chat_sessions(user_id: int) -> list[dict]:
    with closing(get_connection()) as connection:
        connection.execute(
            """
            DELETE FROM chat_sessions
            WHERE user_id = ?
              AND id NOT IN (
                  SELECT DISTINCT chat_id
                  FROM chat_messages
                  WHERE user_id = ? AND chat_id IS NOT NULL
              )
            """,
            (user_id, user_id),
        )
        connection.commit()
        rows = connection.execute(
            """
            SELECT cs.id, cs.title, cs.created_at, cs.last_activity,
                   (SELECT message FROM chat_messages WHERE chat_id = cs.id AND role = 'user' ORDER BY id DESC LIMIT 1) AS last_user_message
            FROM chat_sessions cs
            WHERE cs.user_id = ?
              AND EXISTS (
                  SELECT 1
                  FROM chat_messages cm
                  WHERE cm.user_id = cs.user_id AND cm.chat_id = cs.id
              )
            ORDER BY cs.last_activity DESC, cs.id DESC
            """,
            (user_id,),
        ).fetchall()
    return [
        {
            "id": row["id"],
            "user_id": user_id,
            "title": row["title"],
            "created_at": coerce_iso_timestamp(row["created_at"]),
            "last_activity": coerce_iso_timestamp(row["last_activity"], row["created_at"]),
            "last_user_message": row["last_user_message"],
        }
        for row in rows
    ]


def touch_chat_session(chat_id: int) -> None:
    with closing(get_connection()) as connection:
        connection.execute(
            "UPDATE chat_sessions SET last_activity = ? WHERE id = ?",
            (utc_now_iso(), chat_id),
        )
        connection.commit()


def store_chat_message(
    user_id: int,
    role: str,
    message: str,
    chat_id: int | None = None,
    source: str | None = None,
    confidence: float | None = None,
    metadata: dict | None = None,
) -> None:
    now = utc_now_iso()
    with closing(get_connection()) as connection:
        connection.execute(
            """
            INSERT INTO chat_messages (user_id, chat_id, role, message, source, confidence, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                chat_id,
                role,
                message,
                source,
                confidence,
                json.dumps(metadata or {}),
                now,
            ),
        )
        if chat_id is not None:
            connection.execute(
                "UPDATE chat_sessions SET last_activity = ? WHERE id = ?",
                (now, chat_id),
            )
        connection.commit()


def list_chat_messages(user_id: int, chat_id: int | None = None, limit: int = 100) -> list[dict]:
    with closing(get_connection()) as connection:
        if chat_id is not None:
            rows = connection.execute(
                """
                SELECT id, chat_id, role, message, source, confidence, metadata_json, created_at
                FROM chat_messages
                WHERE user_id = ? AND chat_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, chat_id, limit),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT id, chat_id, role, message, source, confidence, metadata_json, created_at
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
                "chat_id": row["chat_id"],
                "role": row["role"],
                "message": row["message"],
                "source": row["source"],
                "confidence": row["confidence"],
                "metadata": metadata,
                "created_at": row["created_at"],
            }
        )
    return results


def clear_chat_messages(user_id: int, chat_id: int) -> None:
    now = utc_now_iso()
    with closing(get_connection()) as connection:
        matter_rows = connection.execute(
            "SELECT id FROM matter_intakes WHERE user_id = ? AND chat_id = ?",
            (user_id, chat_id),
        ).fetchall()
        matter_ids = [row["id"] for row in matter_rows]
        if matter_ids:
            placeholders = ",".join("?" for _ in matter_ids)
            connection.execute(
                f"DELETE FROM uploaded_documents WHERE user_id = ? AND matter_intake_id IN ({placeholders})",
                (user_id, *matter_ids),
            )
            connection.execute(
                f"DELETE FROM matter_intakes WHERE user_id = ? AND id IN ({placeholders})",
                (user_id, *matter_ids),
            )
        connection.execute(
            "DELETE FROM chat_messages WHERE user_id = ? AND chat_id = ?",
            (user_id, chat_id),
        )
        connection.execute(
            "UPDATE chat_sessions SET last_activity = ? WHERE id = ? AND user_id = ?",
            (now, chat_id, user_id),
        )
        connection.commit()


def clear_all_chat_history(user_id: int) -> None:
    with closing(get_connection()) as connection:
        connection.execute(
            "DELETE FROM uploaded_documents WHERE user_id = ?",
            (user_id,),
        )
        connection.execute(
            "DELETE FROM matter_intakes WHERE user_id = ?",
            (user_id,),
        )
        connection.execute(
            "DELETE FROM chat_messages WHERE user_id = ?",
            (user_id,),
        )
        connection.execute(
            "DELETE FROM chat_sessions WHERE user_id = ?",
            (user_id,),
        )
        connection.commit()

def serialize_matter_intake(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    details = {}

    if row["details_json"]:
        try:
            details = json.loads(row["details_json"])
        except json.JSONDecodeError:
            details = {}

    return {
        "id": row["id"],
        "user_id": row["user_id"],
        "chat_id": row["chat_id"],
        "status": row["status"],
        "initial_query": row["initial_query"],
        "issue_summary": row["issue_summary"],
        "state": row["state"],
        "district": row["district"],
        "matter_type": row["matter_type"],
        "matter_stage": row["matter_stage"],
        "is_own_matter": None if row["is_own_matter"] is None else bool(row["is_own_matter"]),
        "urgency": row["urgency"],
        "last_question_key": row["last_question_key"],
        "relief_goal": row["relief_goal"],
        "details": details,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }

def get_active_matter_intake(user_id: int, chat_id: int | None = None) -> dict | None:
    with closing(get_connection()) as connection:
        if chat_id is None:
            row = connection.execute(
                """
                SELECT *
                FROM matter_intakes
                WHERE user_id = ? AND status = 'active'
                ORDER BY id DESC
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
        else:
            row = connection.execute(
                """
                SELECT *
                FROM matter_intakes
                WHERE user_id = ? AND chat_id = ? AND status = 'active'
                ORDER BY id DESC
                LIMIT 1
                """,
                (user_id, chat_id),
            ).fetchone()
    return serialize_matter_intake(row)

def upsert_active_matter_intake(
    user_id: int,
    *,
    chat_id: int | None = None,
    initial_query: str | None = None,
    issue_summary: str | None = None,
    state: str | None = None,
    district: str | None = None,
    matter_type: str | None = None,
    matter_stage: str | None = None,
    is_own_matter: bool | None = None,
    urgency: str | None = None,
    last_question_key: str | None = None,
    relief_goal: str | None = None,
    details: dict | None = None,
) -> dict:
    existing = get_active_matter_intake(user_id, chat_id=chat_id)
    details_json = json.dumps(details or {})
    now = utc_now_iso()

    with closing(get_connection()) as connection:
        if existing is None:
            cursor = connection.execute(
                """
                INSERT INTO matter_intakes (
                    user_id, chat_id, status, initial_query, issue_summary, state, district,
                    matter_type, matter_stage, is_own_matter, urgency, last_question_key,
                    relief_goal, details_json, created_at, updated_at
                )
                VALUES (?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,

                (
                    user_id,
                    chat_id,
                    initial_query,
                    issue_summary,
                    state,
                    district,
                    matter_type,
                    matter_stage,
                    None if is_own_matter is None else int(is_own_matter),
                    urgency,
                    last_question_key,
                    relief_goal,
                    details_json,
                    now,
                    now,
                ),
            )
            matter_id = cursor.lastrowid

        else:
            matter_id = existing["id"]
            merged_details = existing.get("details") or {}
            merged_details.update(details or {})
            details_json = json.dumps(merged_details)
            connection.execute(
                """
                UPDATE matter_intakes
                SET initial_query = COALESCE(?, initial_query),
                    chat_id = COALESCE(?, chat_id),
                    issue_summary = COALESCE(?, issue_summary),
                    state = COALESCE(?, state),
                    district = COALESCE(?, district),
                    matter_type = COALESCE(?, matter_type),
                    matter_stage = COALESCE(?, matter_stage),
                    is_own_matter = COALESCE(?, is_own_matter),
                    urgency = COALESCE(?, urgency),
                    last_question_key = ?,
                    relief_goal = COALESCE(?, relief_goal),
                    details_json = ?,
                    updated_at = ?
                WHERE id = ?
                """,

                (
                    initial_query,
                    chat_id,
                    issue_summary,
                    state,
                    district,
                    matter_type,
                    matter_stage,
                    None if is_own_matter is None else int(is_own_matter),
                    urgency,
                    last_question_key,
                    relief_goal,
                    details_json,
                    now,
                    matter_id,
                ),
            )

        connection.commit()
        row = connection.execute("SELECT * FROM matter_intakes WHERE id = ?", (matter_id,)).fetchone()
    matter = serialize_matter_intake(row)

    if not matter:
        raise ValueError("Failed to persist matter intake")
    return matter

def close_active_matter_intake(user_id: int, chat_id: int | None = None) -> None:
    with closing(get_connection()) as connection:
        if chat_id is None:
            connection.execute(
                """
                UPDATE matter_intakes
                SET status = 'closed', updated_at = ?
                WHERE user_id = ? AND status = 'active'
                """,
                (utc_now_iso(), user_id),
            )
        else:
            connection.execute(
                """
                UPDATE matter_intakes
                SET status = 'closed', updated_at = ?
                WHERE user_id = ? AND chat_id = ? AND status = 'active'
                """,
                (utc_now_iso(), user_id, chat_id),
            )
        connection.commit()

def store_uploaded_document(
    user_id: int,
    *,
    matter_intake_id: int | None,
    original_name: str,
    stored_name: str,
    file_path: str,
    content_type: str | None,
    file_size: int | None,
    document_kind: str | None,
    description: str | None,
    extracted_text: str | None = None,
) -> dict[str, Any]:

    with closing(get_connection()) as connection:
        cursor = connection.execute(
            """
            INSERT INTO uploaded_documents (
                user_id, matter_intake_id, original_name, stored_name, file_path,
                content_type, file_size, document_kind, description, extracted_text, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                matter_intake_id,
                original_name,
                stored_name,
                file_path,
                content_type,
                file_size,
                document_kind,
                description,
                extracted_text,
                utc_now_iso(),
            ),
        )

        doc_id = cursor.lastrowid
        connection.commit()
        row = connection.execute(
            "SELECT * FROM uploaded_documents WHERE id = ?",
            (doc_id,),
        ).fetchone()
    return serialize_uploaded_document(row)

def serialize_uploaded_document(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    
    return {
        "id": row["id"],
        "user_id": row["user_id"],
        "matter_intake_id": row["matter_intake_id"],
        "original_name": row["original_name"],
        "stored_name": row["stored_name"],
        "file_path": row["file_path"],
        "content_type": row["content_type"],
        "file_size": row["file_size"],
        "document_kind": row["document_kind"],
        "description": row["description"],
        "extracted_text": row["extracted_text"],
        "created_at": row["created_at"],
    }

def list_uploaded_documents(user_id: int, matter_intake_id: int | None = None) -> list[dict[str, Any]]:
    with closing(get_connection()) as connection:
        if matter_intake_id is None:
            rows = connection.execute(
                """
                SELECT *
                FROM uploaded_documents
                WHERE user_id = ?
                ORDER BY id DESC
                """,
                (user_id,),
            ).fetchall()
     
        else:
            rows = connection.execute(
                """
                SELECT *
                FROM uploaded_documents
                WHERE user_id = ? AND matter_intake_id = ?
                ORDER BY id DESC
                """,
                (user_id, matter_intake_id),
            ).fetchall()
    return [serialize_uploaded_document(row) for row in rows if serialize_uploaded_document(row)]
