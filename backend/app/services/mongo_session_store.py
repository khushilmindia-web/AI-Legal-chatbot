from __future__ import annotations

import json
import logging
import secrets
import hashlib
import hmac
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import unquote

from pymongo import ASCENDING, MongoClient, ReturnDocument

from backend.app.core.config import Settings


logger = logging.getLogger(__name__)
PASSWORD_HASH_ITERATIONS = 390000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def session_expiry(days: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()


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


class MongoSessionStore:
    def __init__(self, settings: Settings) -> None:
        self.client: MongoClient = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=5000)
        self.database = self.client[settings.mongodb_database]
        self.users = self.database["users"]
        self.auth_sessions = self.database["auth_sessions"]
        self.chat_sessions = self.database["chat_sessions"]
        self.chat_messages = self.database["chat_messages"]
        self.password_reset_tokens = self.database["password_reset_tokens"]
        self.counters = self.database["counters"]
        self._init_indexes()

    def _init_indexes(self) -> None:
        self.users.create_index([("id", ASCENDING)], unique=True)
        self.users.create_index([("email", ASCENDING)], unique=True)
        self.users.create_index([("google_sub", ASCENDING)], unique=True, sparse=True)
        self.auth_sessions.create_index([("token", ASCENDING)], unique=True)
        self.auth_sessions.create_index([("user_id", ASCENDING)])
        self.auth_sessions.create_index([("expires_at", ASCENDING)])
        self.chat_sessions.create_index([("id", ASCENDING)], unique=True)
        self.chat_sessions.create_index([("user_id", ASCENDING)])
        self.chat_messages.create_index([("id", ASCENDING)], unique=True)
        self.chat_messages.create_index([("chat_id", ASCENDING)])
        self.password_reset_tokens.create_index([("token_hash", ASCENDING)], unique=True)
        self.password_reset_tokens.create_index([("user_id", ASCENDING)])

    def _next_id(self, name: str) -> int:
        item = self.counters.find_one_and_update(
            {"_id": name},
            {"$inc": {"seq": 1}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return int(item["seq"])

    @staticmethod
    def _public_user(document: dict[str, Any] | None) -> dict[str, Any] | None:
        if document is None:
            return None
        return {
            "id": document["id"],
            "full_name": document["full_name"],
            "email": document["email"],
            "state": document.get("state"),
            "created_at": document["created_at"],
        }

    @staticmethod
    def _session_document_to_dict(document: dict[str, Any] | None) -> dict[str, Any] | None:
        if document is None:
            return None
        state = document.get("state")
        if isinstance(state, dict):
            session_state = state
        else:
            raw_state = document.get("state_json")
            if raw_state:
                try:
                    session_state = json.loads(raw_state)
                except json.JSONDecodeError:
                    session_state = {}
            else:
                session_state = {}
        return {
            "id": document["id"],
            "user_id": document.get("user_id"),
            "title": document["title"],
            "state": session_state,
            "created_at": document["created_at"],
            "updated_at": document["updated_at"],
        }

    def create_user(self, full_name: str, email: str, password: str, state: str | None = None) -> dict[str, Any]:
        normalized_email = email.strip().lower()
        now = utc_now()
        user_id = self._next_id("users")
        document = {
            "id": user_id,
            "full_name": full_name.strip(),
            "email": normalized_email,
            "password_hash": hash_password(password),
            "auth_provider": "local",
            "state": state,
            "created_at": now,
        }
        self.users.insert_one(document)
        user = self._public_user(document)
        if user is None:
            raise ValueError("Failed to create user")
        return user

    def get_user_by_email(self, email: str) -> dict[str, Any] | None:
        normalized_email = email.strip().lower()
        return self._public_user(self.users.find_one({"email": normalized_email}))

    def get_user_by_google_sub(self, google_sub: str) -> dict[str, Any] | None:
        user = self._public_user(self.users.find_one({"google_sub": google_sub}))
        logger.info("mongodb google user lookup by_sub result=%s", "hit" if user else "miss")
        return user

    def authenticate_user(self, email: str, password: str) -> dict[str, Any] | None:
        normalized_email = email.strip().lower()
        document = self.users.find_one({"email": normalized_email})
        if document is None or not verify_password(password, document.get("password_hash", "")):
            return None
        return self._public_user(document)

    def create_google_user(self, full_name: str, email: str, google_sub: str, state: str | None = None) -> dict[str, Any]:
        normalized_email = email.strip().lower()
        now = utc_now()
        user_id = self._next_id("users")
        document = {
            "id": user_id,
            "full_name": full_name.strip(),
            "email": normalized_email,
            "password_hash": hash_password(secrets.token_urlsafe(32)),
            "auth_provider": "google",
            "google_sub": google_sub,
            "state": state,
            "created_at": now,
        }
        self.users.insert_one(document)
        logger.info("mongodb google user created user_id=%s email=%s", user_id, normalized_email)
        user = self._public_user(document)
        if user is None:
            raise ValueError("Failed to create Google user")
        return user

    def link_google_account(self, user_id: int, google_sub: str) -> dict[str, Any] | None:
        current = self.users.find_one({"id": user_id})
        if current is None:
            return None
        next_provider = "both" if current.get("auth_provider") == "local" else current.get("auth_provider", "google")
        self.users.update_one(
            {"id": user_id},
            {"$set": {"google_sub": google_sub, "auth_provider": next_provider}},
        )
        logger.info("mongodb google user linked user_id=%s provider=%s", user_id, next_provider)
        return self._public_user(self.users.find_one({"id": user_id}))

    def create_auth_session(self, user_id: int, duration_days: int = 14) -> str:
        token = secrets.token_urlsafe(32)
        session_id = self._next_id("auth_sessions")
        self.auth_sessions.insert_one(
            {
                "id": session_id,
                "user_id": user_id,
                "token": token,
                "created_at": utc_now(),
                "expires_at": session_expiry(duration_days),
            }
        )
        logger.info("mongodb auth_session created session_id=%s user_id=%s token_prefix=%s", session_id, user_id, token[:8])
        return token

    def get_user_by_token(self, token: str) -> dict[str, Any] | None:
        if not token:
            return None
        session_document = self.auth_sessions.find_one({"token": token})
        if session_document is None:
            logger.info("mongodb auth_session lookup result=miss token_prefix=%s", token[:8])
            return None
        expires_at = datetime.fromisoformat(session_document["expires_at"])
        if expires_at < datetime.now(timezone.utc):
            self.auth_sessions.delete_one({"token": token})
            logger.info("mongodb auth_session lookup result=expired session_id=%s user_id=%s", session_document.get("id"), session_document.get("user_id"))
            return None
        user = self._public_user(self.users.find_one({"id": session_document["user_id"]}))
        logger.info(
            "mongodb auth_session lookup result=%s session_id=%s user_id=%s token_prefix=%s",
            "hit" if user else "user_missing",
            session_document.get("id"),
            session_document.get("user_id"),
            token[:8],
        )
        return user

    def delete_auth_session(self, token: str) -> None:
        if token:
            result = self.auth_sessions.delete_one({"token": token})
            logger.info("mongodb auth_session deleted count=%s token_prefix=%s", result.deleted_count, token[:8])

    def create_password_reset_token(self, user_id: int, ttl_minutes: int = 30) -> str:
        token = secrets.token_urlsafe(32)
        now = utc_now()
        expires_at = (datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)).isoformat()
        token_digest = hash_token(token)
        self.password_reset_tokens.delete_many({"user_id": user_id})
        self.password_reset_tokens.insert_one(
            {
                "id": self._next_id("password_reset_tokens"),
                "user_id": user_id,
                "token_hash": token_digest,
                "created_at": now,
                "expires_at": expires_at,
                "used_at": None,
            }
        )
        logger.info(
            "Created password reset token user_id=%s token_hash_prefix=%s expires_at=%s db=mongodb",
            user_id,
            token_digest[:12],
            expires_at,
        )
        return token

    def reset_password_with_token(self, token: str, new_password: str) -> dict[str, Any] | None:
        normalized_token = normalize_token(token)
        token_digest = hash_token(normalized_token)
        now = datetime.now(timezone.utc)
        used_at = now.isoformat()
        document = self.password_reset_tokens.find_one({"token_hash": token_digest})
        if document is None or document.get("used_at"):
            return None
        expires_at = datetime.fromisoformat(document["expires_at"])
        if expires_at < now:
            return None
        user_document = self.users.find_one({"id": document["user_id"]})
        if user_document is None:
            return None
        next_provider = "both" if user_document.get("google_sub") else "local"
        self.users.update_one(
            {"id": document["user_id"]},
            {"$set": {"password_hash": hash_password(new_password), "auth_provider": next_provider}},
        )
        self.password_reset_tokens.update_one({"token_hash": token_digest}, {"$set": {"used_at": used_at}})
        self.auth_sessions.delete_many({"user_id": document["user_id"]})
        return self._public_user(self.users.find_one({"id": document["user_id"]}))

    def create_session(self, title: str, user_id: int | None = None) -> dict[str, Any]:
        now = utc_now()
        chat_id = self._next_id("chat_sessions")
        document = {
            "id": chat_id,
            "user_id": user_id,
            "title": title,
            "state": {},
            "created_at": now,
            "updated_at": now,
        }
        self.chat_sessions.insert_one(document)
        return self._session_document_to_dict(document) or {}

    def get_session(self, chat_id: int, user_id: int | None = None) -> dict[str, Any] | None:
        query: dict[str, Any] = {"id": chat_id}
        if user_id is not None:
            query["user_id"] = user_id
        return self._session_document_to_dict(self.chat_sessions.find_one(query))

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
        query: dict[str, Any] = {"id": chat_id}
        if user_id is not None:
            query["user_id"] = user_id
        result = self.chat_sessions.update_one(
            query,
            {"$set": {"state": state or {}, "updated_at": utc_now()}},
        )
        if user_id is not None and result.matched_count == 0:
            raise ValueError("Chat session not found")
        return state

    def add_message(
        self,
        chat_id: int,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        user_id: int | None = None,
    ) -> dict[str, Any]:
        if user_id is not None and self.chat_sessions.find_one({"id": chat_id, "user_id": user_id}) is None:
            raise ValueError("Chat session not found")
        now = utc_now()
        message_id = self._next_id("chat_messages")
        document = {
            "id": message_id,
            "chat_id": chat_id,
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "created_at": now,
        }
        self.chat_messages.insert_one(document)
        self.chat_sessions.update_one({"id": chat_id}, {"$set": {"updated_at": now}})
        return dict(document)

    def list_sessions(self, user_id: int | None = None) -> list[dict[str, Any]]:
        query: dict[str, Any] = {}
        if user_id is not None:
            query["user_id"] = user_id
        rows: list[dict[str, Any]] = []
        for session in self.chat_sessions.find(query).sort([("updated_at", -1), ("id", -1)]):
            last_user_message = self.chat_messages.find_one(
                {"chat_id": session["id"], "role": "user"},
                sort=[("id", -1)],
            )
            if last_user_message is None:
                continue
            rows.append(
                {
                    "id": session["id"],
                    "user_id": session.get("user_id"),
                    "title": session["title"],
                    "created_at": session["created_at"],
                    "updated_at": session["updated_at"],
                    "last_message_preview": last_user_message.get("content"),
                }
            )
        return rows

    def get_messages(self, chat_id: int, user_id: int | None = None) -> list[dict[str, Any]]:
        if user_id is not None and self.chat_sessions.find_one({"id": chat_id, "user_id": user_id}) is None:
            return []
        items: list[dict[str, Any]] = []
        for document in self.chat_messages.find({"chat_id": chat_id}).sort("id", ASCENDING):
            metadata = document.get("metadata")
            if metadata is None and document.get("metadata_json"):
                try:
                    metadata = json.loads(document["metadata_json"])
                except json.JSONDecodeError:
                    metadata = {}
            items.append(
                {
                    "id": document["id"],
                    "chat_id": document["chat_id"],
                    "role": document["role"],
                    "content": document["content"],
                    "metadata": metadata or {},
                    "created_at": document["created_at"],
                }
            )
        return items

    def clear_history(self, user_id: int | None = None) -> None:
        if user_id is None:
            self.chat_messages.delete_many({})
            self.chat_sessions.delete_many({})
            return
        session_ids = [item["id"] for item in self.chat_sessions.find({"user_id": user_id}, {"id": 1})]
        if session_ids:
            self.chat_messages.delete_many({"chat_id": {"$in": session_ids}})
        self.chat_sessions.delete_many({"user_id": user_id})

    def delete_session(self, chat_id: int, user_id: int | None = None) -> bool:
        query: dict[str, Any] = {"id": chat_id}
        if user_id is not None:
            query["user_id"] = user_id
        result = self.chat_sessions.delete_one(query)
        if result.deleted_count:
            self.chat_messages.delete_many({"chat_id": chat_id})
            return True
        return False
