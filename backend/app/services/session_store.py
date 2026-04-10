from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import sqlite3
import tempfile
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote


PASSWORD_HASH_ITERATIONS = 390000
logger = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def session_expiry(days: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()


def column_exists(connection: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(row["name"] == column_name for row in rows)


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_HASH_ITERATIONS)
    return f"{PASSWORD_HASH_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        iterations_raw, salt_hex, digest_hex = stored_hash.split("$", 2)
        iterations = int(iterations_raw)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except (ValueError, TypeError):
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(actual, expected)


def normalize_token(token: str) -> str:
    normalized = str(token or "").strip().strip("\"'")
    for _ in range(2):
        decoded = unquote(normalized)
        if decoded == normalized:
            break
        normalized = decoded.strip().strip("\"'")
    return normalized


def hash_token(token: str) -> str:
    normalized = normalize_token(token)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def serialize_user(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "id": row["id"],
        "full_name": row["full_name"],
        "email": row["email"],
        "state": row["state"],
        "created_at": row["created_at"],
    }


class SessionStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._init_db()
        except sqlite3.OperationalError as exc:
            logger.warning("Primary sqlite init failed for %s; switching to temp fallback: %s", self.db_path, exc)
            fallback_path = self._build_fallback_db_path(self.db_path)
            self.db_path = fallback_path
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

    @staticmethod
    def _build_fallback_db_path(original_path: Path) -> Path:
        resolved = str(original_path.resolve())
        digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:12]
        stem = original_path.stem or "lawyer_ai"
        suffix = original_path.suffix or ".db"
        return Path(tempfile.gettempdir()) / f"{stem}_runtime_{digest}{suffix}"

    def _get_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _init_db(self) -> None:
        with closing(self._get_connection()) as connection:
            connection.executescript(
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
                );

                CREATE TABLE IF NOT EXISTS auth_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    token TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    title TEXT NOT NULL,
                    state_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(chat_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS password_reset_tokens (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    token_hash TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    used_at TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                """
            )
            if not column_exists(connection, "users", "auth_provider"):
                connection.execute("ALTER TABLE users ADD COLUMN auth_provider TEXT NOT NULL DEFAULT 'local'")
            if not column_exists(connection, "users", "google_sub"):
                connection.execute("ALTER TABLE users ADD COLUMN google_sub TEXT")
            if not column_exists(connection, "chat_sessions", "user_id"):
                connection.execute("ALTER TABLE chat_sessions ADD COLUMN user_id INTEGER")
            if not column_exists(connection, "chat_sessions", "state_json"):
                connection.execute("ALTER TABLE chat_sessions ADD COLUMN state_json TEXT")
            connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email)")
            connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_google_sub ON users(google_sub)")
            connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_auth_sessions_token ON auth_sessions(token)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_chat_id ON chat_messages(chat_id)")
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_password_reset_tokens_hash ON password_reset_tokens(token_hash)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_password_reset_tokens_user_id ON password_reset_tokens(user_id)"
            )
            connection.commit()

    def create_user(self, full_name: str, email: str, password: str, state: str | None = None) -> dict[str, Any]:
        normalized_email = email.strip().lower()
        now = utc_now()
        with closing(self._get_connection()) as connection:
            cursor = connection.execute(
                """
                INSERT INTO users (full_name, email, password_hash, auth_provider, state, created_at)
                VALUES (?, ?, ?, 'local', ?, ?)
                """,
                (full_name.strip(), normalized_email, hash_password(password), state, now),
            )
            row = connection.execute("SELECT * FROM users WHERE id = ?", (cursor.lastrowid,)).fetchone()
            connection.commit()
        user = serialize_user(row)
        if user is None:
            raise ValueError("Failed to create user")
        return user

    def get_user_by_email(self, email: str) -> dict[str, Any] | None:
        normalized_email = email.strip().lower()
        with closing(self._get_connection()) as connection:
            row = connection.execute("SELECT * FROM users WHERE email = ?", (normalized_email,)).fetchone()
        return serialize_user(row)

    def get_user_by_google_sub(self, google_sub: str) -> dict[str, Any] | None:
        with closing(self._get_connection()) as connection:
            row = connection.execute("SELECT * FROM users WHERE google_sub = ?", (google_sub,)).fetchone()
        return serialize_user(row)

    def authenticate_user(self, email: str, password: str) -> dict[str, Any] | None:
        normalized_email = email.strip().lower()
        with closing(self._get_connection()) as connection:
            row = connection.execute("SELECT * FROM users WHERE email = ?", (normalized_email,)).fetchone()
        if row is None or not verify_password(password, row["password_hash"]):
            return None
        return serialize_user(row)

    def create_google_user(self, full_name: str, email: str, google_sub: str, state: str | None = None) -> dict[str, Any]:
        normalized_email = email.strip().lower()
        now = utc_now()
        placeholder_password = hash_password(secrets.token_urlsafe(32))
        with closing(self._get_connection()) as connection:
            cursor = connection.execute(
                """
                INSERT INTO users (full_name, email, password_hash, auth_provider, google_sub, state, created_at)
                VALUES (?, ?, ?, 'google', ?, ?, ?)
                """,
                (full_name.strip(), normalized_email, placeholder_password, google_sub, state, now),
            )
            row = connection.execute("SELECT * FROM users WHERE id = ?", (cursor.lastrowid,)).fetchone()
            connection.commit()
        user = serialize_user(row)
        if user is None:
            raise ValueError("Failed to create Google user")
        return user

    def link_google_account(self, user_id: int, google_sub: str) -> dict[str, Any] | None:
        with closing(self._get_connection()) as connection:
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
            row = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            connection.commit()
        return serialize_user(row)

    def create_auth_session(self, user_id: int, duration_days: int = 14) -> str:
        token = secrets.token_urlsafe(32)
        now = utc_now()
        expires_at = session_expiry(duration_days)
        with closing(self._get_connection()) as connection:
            connection.execute(
                """
                INSERT INTO auth_sessions (user_id, token, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, token, now, expires_at),
            )
            connection.commit()
        return token

    def get_user_by_token(self, token: str) -> dict[str, Any] | None:
        if not token:
            return None
        with closing(self._get_connection()) as connection:
            session_row = connection.execute(
                """
                SELECT user_id, expires_at
                FROM auth_sessions
                WHERE token = ?
                """,
                (token,),
            ).fetchone()
            if session_row is None:
                return None
            expires_at = datetime.fromisoformat(session_row["expires_at"])
            if expires_at < datetime.now(timezone.utc):
                connection.execute("DELETE FROM auth_sessions WHERE token = ?", (token,))
                connection.commit()
                return None
            user_row = connection.execute("SELECT * FROM users WHERE id = ?", (session_row["user_id"],)).fetchone()
        return serialize_user(user_row)

    def delete_auth_session(self, token: str) -> None:
        if not token:
            return
        with closing(self._get_connection()) as connection:
            connection.execute("DELETE FROM auth_sessions WHERE token = ?", (token,))
            connection.commit()

    def create_password_reset_token(self, user_id: int, ttl_minutes: int = 30) -> str:
        token = secrets.token_urlsafe(32)
        now = utc_now()
        expires_at = (datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)).isoformat()
        token_digest = hash_token(token)
        with closing(self._get_connection()) as connection:
            connection.execute("DELETE FROM password_reset_tokens WHERE user_id = ?", (user_id,))
            connection.execute(
                """
                INSERT INTO password_reset_tokens (user_id, token_hash, created_at, expires_at, used_at)
                VALUES (?, ?, ?, ?, NULL)
                """,
                (user_id, token_digest, now, expires_at),
            )
            connection.commit()
        logger.info(
            "Created password reset token user_id=%s token_hash_prefix=%s expires_at=%s db_path=%s",
            user_id,
            token_digest[:12],
            expires_at,
            self.db_path,
        )
        return token

    def reset_password_with_token(self, token: str, new_password: str) -> dict[str, Any] | None:
        normalized_token = normalize_token(token)
        token_digest = hash_token(normalized_token)
        now = datetime.now(timezone.utc)
        used_at = now.isoformat()
        with closing(self._get_connection()) as connection:
            row = connection.execute(
                """
                SELECT user_id, expires_at, used_at
                FROM password_reset_tokens
                WHERE token_hash = ?
                """,
                (token_digest,),
            ).fetchone()
            if row is None:
                logger.warning(
                    "Password reset rejected: token not found token_hash_prefix=%s db_path=%s",
                    token_digest[:12],
                    self.db_path,
                )
                return None
            if row["used_at"]:
                logger.warning(
                    "Password reset rejected: token already used user_id=%s token_hash_prefix=%s used_at=%s",
                    row["user_id"],
                    token_digest[:12],
                    row["used_at"],
                )
                return None
            expires_at = datetime.fromisoformat(row["expires_at"])
            if expires_at < now:
                logger.warning(
                    "Password reset rejected: token expired user_id=%s token_hash_prefix=%s expires_at=%s now=%s",
                    row["user_id"],
                    token_digest[:12],
                    row["expires_at"],
                    used_at,
                )
                return None
            connection.execute(
                """
                UPDATE users
                SET password_hash = ?,
                    auth_provider = CASE
                        WHEN google_sub IS NULL THEN 'local'
                        ELSE 'both'
                    END
                WHERE id = ?
                """,
                (hash_password(new_password), row["user_id"]),
            )
            connection.execute(
                "UPDATE password_reset_tokens SET used_at = ? WHERE token_hash = ?",
                (used_at, token_digest),
            )
            connection.execute("DELETE FROM auth_sessions WHERE user_id = ?", (row["user_id"],))
            user_row = connection.execute("SELECT * FROM users WHERE id = ?", (row["user_id"],)).fetchone()
            connection.commit()
        logger.info(
            "Password reset succeeded user_id=%s token_hash_prefix=%s",
            row["user_id"],
            token_digest[:12],
        )
        return serialize_user(user_row)

    def create_session(self, title: str, user_id: int | None = None) -> dict[str, Any]:
        now = utc_now()
        empty_state = json.dumps({})
        with closing(self._get_connection()) as connection:
            cursor = connection.execute(
                "INSERT INTO chat_sessions (user_id, title, state_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, title, empty_state, now, now),
            )
            connection.commit()
            return {
                "id": cursor.lastrowid,
                "user_id": user_id,
                "title": title,
                "state": {},
                "created_at": now,
                "updated_at": now,
            }

    def get_session(self, chat_id: int, user_id: int | None = None) -> dict[str, Any] | None:
        with closing(self._get_connection()) as connection:
            if user_id is None:
                row = connection.execute(
                    "SELECT id, user_id, title, state_json, created_at, updated_at FROM chat_sessions WHERE id = ?",
                    (chat_id,),
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT id, user_id, title, state_json, created_at, updated_at
                    FROM chat_sessions
                    WHERE id = ? AND user_id = ?
                    """,
                    (chat_id, user_id),
                ).fetchone()
        if row is None:
            return None
        item = dict(row)
        state = {}
        raw_state = item.pop("state_json", None)
        if raw_state:
            try:
                state = json.loads(raw_state)
            except json.JSONDecodeError:
                state = {}
        item["state"] = state
        return item

    def get_conversation_state(self, chat_id: int, user_id: int | None = None) -> dict[str, Any]:
        session = self.get_session(chat_id, user_id=user_id)
        if session is None:
            return {}
        state = session.get("state")
        return state if isinstance(state, dict) else {}

    def update_conversation_state(
        self,
        chat_id: int,
        state: dict[str, Any],
        user_id: int | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        payload = json.dumps(state or {})
        with closing(self._get_connection()) as connection:
            if user_id is not None:
                owns_session = connection.execute(
                    "SELECT 1 FROM chat_sessions WHERE id = ? AND user_id = ?",
                    (chat_id, user_id),
                ).fetchone()
                if owns_session is None:
                    raise ValueError("Chat session not found")
            connection.execute(
                "UPDATE chat_sessions SET state_json = ?, updated_at = ? WHERE id = ?",
                (payload, now, chat_id),
            )
            connection.commit()
        return state

    def add_message(
        self,
        chat_id: int,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        user_id: int | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        with closing(self._get_connection()) as connection:
            if user_id is not None:
                owns_session = connection.execute(
                    "SELECT 1 FROM chat_sessions WHERE id = ? AND user_id = ?",
                    (chat_id, user_id),
                ).fetchone()
                if owns_session is None:
                    raise ValueError("Chat session not found")
            cursor = connection.execute(
                """
                INSERT INTO chat_messages (chat_id, role, content, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (chat_id, role, content, json.dumps(metadata or {}), now),
            )
            connection.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
                (now, chat_id),
            )
            connection.commit()
            return {
                "id": cursor.lastrowid,
                "chat_id": chat_id,
                "role": role,
                "content": content,
                "metadata": metadata or {},
                "created_at": now,
            }

    def list_sessions(self, user_id: int | None = None) -> list[dict[str, Any]]:
        with closing(self._get_connection()) as connection:
            if user_id is None:
                rows = connection.execute(
                    """
                    SELECT s.id, s.user_id, s.title, s.created_at, s.updated_at,
                           (
                               SELECT content
                               FROM chat_messages m
                               WHERE m.chat_id = s.id AND m.role = 'user'
                               ORDER BY m.id DESC
                               LIMIT 1
                           ) AS last_message_preview
                    FROM chat_sessions s
                    WHERE EXISTS (
                        SELECT 1 FROM chat_messages m2 WHERE m2.chat_id = s.id
                    )
                    ORDER BY s.updated_at DESC, s.id DESC
                    """
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT s.id, s.user_id, s.title, s.created_at, s.updated_at,
                           (
                               SELECT content
                               FROM chat_messages m
                               WHERE m.chat_id = s.id AND m.role = 'user'
                               ORDER BY m.id DESC
                               LIMIT 1
                           ) AS last_message_preview
                    FROM chat_sessions s
                    WHERE s.user_id = ?
                      AND EXISTS (
                          SELECT 1 FROM chat_messages m2 WHERE m2.chat_id = s.id
                      )
                    ORDER BY s.updated_at DESC, s.id DESC
                    """,
                    (user_id,),
                ).fetchall()
        return [dict(row) for row in rows]

    def get_messages(self, chat_id: int, user_id: int | None = None) -> list[dict[str, Any]]:
        with closing(self._get_connection()) as connection:
            if user_id is not None:
                owns_session = connection.execute(
                    "SELECT 1 FROM chat_sessions WHERE id = ? AND user_id = ?",
                    (chat_id, user_id),
                ).fetchone()
                if owns_session is None:
                    return []
            rows = connection.execute(
                """
                SELECT id, chat_id, role, content, metadata_json, created_at
                FROM chat_messages
                WHERE chat_id = ?
                ORDER BY id ASC
                """,
                (chat_id,),
            ).fetchall()
        items: list[dict[str, Any]] = []
        for row in rows:
            metadata = {}
            if row["metadata_json"]:
                try:
                    metadata = json.loads(row["metadata_json"])
                except json.JSONDecodeError:
                    metadata = {}
            items.append(
                {
                    "id": row["id"],
                    "chat_id": row["chat_id"],
                    "role": row["role"],
                    "content": row["content"],
                    "metadata": metadata,
                    "created_at": row["created_at"],
                }
            )
        return items

    def clear_history(self, user_id: int | None = None) -> None:
        with closing(self._get_connection()) as connection:
            if user_id is None:
                connection.execute("DELETE FROM chat_messages")
                connection.execute("DELETE FROM chat_sessions")
            else:
                connection.execute(
                    """
                    DELETE FROM chat_messages
                    WHERE chat_id IN (SELECT id FROM chat_sessions WHERE user_id = ?)
                    """,
                    (user_id,),
                )
                connection.execute("DELETE FROM chat_sessions WHERE user_id = ?", (user_id,))
            connection.commit()

    def delete_session(self, chat_id: int, user_id: int | None = None) -> bool:
        with closing(self._get_connection()) as connection:
            if user_id is None:
                cursor = connection.execute("DELETE FROM chat_sessions WHERE id = ?", (chat_id,))
            else:
                cursor = connection.execute(
                    "DELETE FROM chat_sessions WHERE id = ? AND user_id = ?",
                    (chat_id, user_id),
                )
            connection.commit()
            return cursor.rowcount > 0
