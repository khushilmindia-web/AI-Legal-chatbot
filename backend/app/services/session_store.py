from __future__ import annotations

import json
import sqlite3
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._init_db()
        except sqlite3.OperationalError:
            fallback_path = Path(tempfile.gettempdir()) / f"{self.db_path.stem}_runtime{self.db_path.suffix}"
            self.db_path = fallback_path
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

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
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
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
                """
            )
            connection.commit()

    def create_session(self, title: str) -> dict[str, Any]:
        now = utc_now()
        with closing(self._get_connection()) as connection:
            cursor = connection.execute(
                "INSERT INTO chat_sessions (title, created_at, updated_at) VALUES (?, ?, ?)",
                (title, now, now),
            )
            connection.commit()
            return {
                "id": cursor.lastrowid,
                "title": title,
                "created_at": now,
                "updated_at": now,
            }

    def add_message(self, chat_id: int, role: str, content: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        now = utc_now()
        with closing(self._get_connection()) as connection:
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

    def list_sessions(self) -> list[dict[str, Any]]:
        with closing(self._get_connection()) as connection:
            rows = connection.execute(
                """
                SELECT s.id, s.title, s.created_at, s.updated_at,
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
        return [dict(row) for row in rows]

    def get_messages(self, chat_id: int) -> list[dict[str, Any]]:
        with closing(self._get_connection()) as connection:
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

    def clear_history(self) -> None:
        with closing(self._get_connection()) as connection:
            connection.execute("DELETE FROM chat_messages")
            connection.execute("DELETE FROM chat_sessions")
            connection.commit()
