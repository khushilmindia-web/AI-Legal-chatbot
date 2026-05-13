from __future__ import annotations

import json
import logging
import secrets
import hashlib
import hmac
import re
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
        self.message_feedback = self.database["message_feedback"]
        self.admin_activity = self.database["admin_activity"]
        self.runtime_config = self.database["runtime_config"]
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
        self.message_feedback.create_index([("id", ASCENDING)], unique=True)
        self.message_feedback.create_index([("message_id", ASCENDING)])
        self.message_feedback.create_index([("chat_id", ASCENDING)])
        self.message_feedback.create_index([("user_id", ASCENDING)])
        self.admin_activity.create_index([("id", ASCENDING)], unique=True)
        self.admin_activity.create_index([("admin_user_id", ASCENDING)])
        self.admin_activity.create_index([("created_at", ASCENDING)])
        self.runtime_config.create_index([("key", ASCENDING)], unique=True)
        self.password_reset_tokens.create_index([("token_hash", ASCENDING)], unique=True)
        self.password_reset_tokens.create_index([("user_id", ASCENDING)])
        self.users.update_many({"status": {"$nin": ["active", "blocked"]}}, {"$set": {"status": "active"}})

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
            "role": document.get("role") or "user",
            "status": document.get("status") if document.get("status") in {"active", "blocked"} else "active",
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
            "role": "user",
            "status": "active",
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

    def get_user_by_id(self, user_id: int) -> dict[str, Any] | None:
        return self._public_user(self.users.find_one({"id": int(user_id)}))

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
            "role": "user",
            "status": "active",
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

    def promote_user_to_admin(self, email: str) -> dict[str, Any] | None:
        normalized_email = email.strip().lower()
        result = self.users.update_one(
            {"email": normalized_email},
            {"$set": {"role": "admin"}},
        )
        if result.matched_count == 0:
            return None
        logger.info("mongodb user promoted to admin email=%s", normalized_email)
        return self._public_user(self.users.find_one({"email": normalized_email}))

    def update_user_status(self, user_id: int, status: str) -> dict[str, Any] | None:
        normalized_status = str(status or "").strip().lower()
        if normalized_status not in {"active", "blocked"}:
            raise ValueError("User status must be active or blocked")
        result = self.users.update_one(
            {"id": int(user_id)},
            {"$set": {"status": normalized_status}},
        )
        if result.matched_count == 0:
            return None
        logger.info("mongodb user status updated user_id=%s status=%s", user_id, normalized_status)
        return self._public_user(self.users.find_one({"id": int(user_id)}))

    def get_admin_stats(self) -> dict[str, Any]:
        recent_sessions = []
        for session in self.chat_sessions.find({}).sort([("updated_at", -1), ("id", -1)]).limit(5):
            recent_sessions.append(
                {
                    "id": session["id"],
                    "user_id": session.get("user_id"),
                    "title": session.get("title", "Untitled chat"),
                    "created_at": session["created_at"],
                    "updated_at": session["updated_at"],
                }
            )
        return {
            "total_users": self.users.count_documents({}),
            "total_chat_sessions": self.chat_sessions.count_documents({}),
            "total_messages": self.chat_messages.count_documents({}),
            "recent_sessions": recent_sessions,
        }

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime | None:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        if isinstance(value, str) and value.strip():
            try:
                parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
            except ValueError:
                return None
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        return None

    @staticmethod
    def _period_start(period: str) -> tuple[str, datetime]:
        normalized = str(period or "24h").strip().lower()
        if normalized not in {"24h", "7d", "30d"}:
            normalized = "24h"
        days = 1 if normalized == "24h" else 7 if normalized == "7d" else 30
        return normalized, datetime.now(timezone.utc) - timedelta(days=days)

    @staticmethod
    def _numeric_confidence(metadata: dict[str, Any]) -> float | None:
        confidence = metadata.get("confidence")
        if confidence is None:
            retrieval = metadata.get("retrieval")
            if isinstance(retrieval, dict):
                confidence = retrieval.get("retrieval_confidence") or retrieval.get("confidence")
        if isinstance(confidence, (int, float)):
            return float(confidence)
        try:
            return float(str(confidence).strip())
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _is_fallback_metadata(metadata: dict[str, Any]) -> bool:
        safe_metadata = MongoSessionStore._admin_metadata(metadata)
        fallback_reason = str(safe_metadata.get("fallback_reason") or "").strip().lower()
        source = str(safe_metadata.get("source") or "").strip().lower()
        return bool(fallback_reason) or "fallback" in source

    @staticmethod
    def _is_unsupported_metadata(metadata: dict[str, Any]) -> bool:
        safe_metadata = MongoSessionStore._admin_metadata(metadata)
        fallback_reason = str(safe_metadata.get("fallback_reason") or "").strip().lower()
        validation_flags = safe_metadata.get("validation_flags")
        if isinstance(validation_flags, list):
            validation_text = " ".join(str(flag) for flag in validation_flags).lower()
        else:
            validation_text = str(validation_flags or "").lower()
        return "unsupported_output" in fallback_reason or "unsupported_output" in validation_text

    def get_admin_analytics_overview(self, *, period: str = "24h", average_response_time_ms: float | None = None) -> dict[str, Any]:
        normalized_period, started_at = self._period_start(period)
        ended_at = datetime.now(timezone.utc)
        total_chats = 0
        active_user_ids: set[int] = set()
        for session in self.chat_sessions.find({}, {"id": 1, "user_id": 1, "created_at": 1, "updated_at": 1}):
            created_at = self._parse_timestamp(session.get("created_at"))
            updated_at = self._parse_timestamp(session.get("updated_at"))
            if created_at is not None and created_at >= started_at:
                total_chats += 1
            if updated_at is not None and updated_at >= started_at and session.get("user_id") is not None:
                try:
                    active_user_ids.add(int(session["user_id"]))
                except (TypeError, ValueError):
                    pass

        total_messages = 0
        fallback_count = 0
        unsupported_response_count = 0
        confidence_values: list[float] = []
        for message in self.chat_messages.find({}, {"role": 1, "metadata": 1, "created_at": 1}):
            created_at = self._parse_timestamp(message.get("created_at"))
            if created_at is None or created_at < started_at:
                continue
            total_messages += 1
            if message.get("role") != "assistant":
                continue
            metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
            if self._is_fallback_metadata(metadata):
                fallback_count += 1
            if self._is_unsupported_metadata(metadata):
                unsupported_response_count += 1
            confidence = self._numeric_confidence(metadata)
            if confidence is not None:
                confidence_values.append(confidence)

        average_confidence = round(sum(confidence_values) / len(confidence_values), 3) if confidence_values else None
        return {
            "period": normalized_period,
            "generated_at": ended_at.isoformat(),
            "window": {
                "started_at": started_at.isoformat(),
                "ended_at": ended_at.isoformat(),
            },
            "metrics": {
                "total_chats": total_chats,
                "total_messages": total_messages,
                "active_users": len(active_user_ids),
                "fallback_count": fallback_count,
                "unsupported_response_count": unsupported_response_count,
                "average_confidence": average_confidence,
                "average_response_time_ms": average_response_time_ms,
            },
        }

    @staticmethod
    def _trend_period_days(period: str) -> tuple[str, int]:
        normalized = str(period or "7d").strip().lower()
        if normalized not in {"7d", "30d"}:
            normalized = "7d"
        return normalized, 7 if normalized == "7d" else 30

    def get_admin_analytics_trends(self, *, period: str = "7d") -> dict[str, Any]:
        normalized_period, days = self._trend_period_days(period)
        ended_at = datetime.now(timezone.utc)
        start_date = (ended_at.date() - timedelta(days=days - 1))
        buckets: dict[str, dict[str, Any]] = {}
        for offset in range(days):
            day = start_date + timedelta(days=offset)
            day_key = day.isoformat()
            buckets[day_key] = {
                "date": day_key,
                "chats": 0,
                "messages": 0,
                "fallback_count": 0,
                "average_confidence": None,
                "_confidence_values": [],
            }

        for session in self.chat_sessions.find({}, {"created_at": 1}):
            created_at = self._parse_timestamp(session.get("created_at"))
            if created_at is None:
                continue
            day_key = created_at.date().isoformat()
            if day_key in buckets:
                buckets[day_key]["chats"] += 1

        for message in self.chat_messages.find({}, {"role": 1, "metadata": 1, "created_at": 1}):
            created_at = self._parse_timestamp(message.get("created_at"))
            if created_at is None:
                continue
            day_key = created_at.date().isoformat()
            if day_key not in buckets:
                continue
            buckets[day_key]["messages"] += 1
            if message.get("role") != "assistant":
                continue
            metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
            if self._is_fallback_metadata(metadata):
                buckets[day_key]["fallback_count"] += 1
            confidence = self._numeric_confidence(metadata)
            if confidence is not None:
                buckets[day_key]["_confidence_values"].append(confidence)

        items = []
        for day_key in sorted(buckets):
            bucket = buckets[day_key]
            confidence_values = bucket.pop("_confidence_values")
            bucket["average_confidence"] = (
                round(sum(confidence_values) / len(confidence_values), 3)
                if confidence_values
                else None
            )
            items.append(bucket)

        return {
            "period": normalized_period,
            "generated_at": ended_at.isoformat(),
            "items": items,
        }

    @staticmethod
    def _category_from_metadata(metadata: dict[str, Any]) -> str:
        if not isinstance(metadata, dict):
            return "other"
        retrieval = metadata.get("retrieval")
        if not isinstance(retrieval, dict):
            retrieval = {}
        query_profile = metadata.get("query_profile")
        if not isinstance(query_profile, dict):
            query_profile = retrieval.get("query_profile") if isinstance(retrieval.get("query_profile"), dict) else {}
        values: list[str] = []
        for source in (metadata, retrieval, query_profile):
            if not isinstance(source, dict):
                continue
            for key in (
                "domain",
                "legal_domain",
                "route_type",
                "pipeline",
                "pipeline_path",
                "source",
                "answer_mode",
                "query_type",
                "playbook_id",
                "route_target",
            ):
                value = source.get(key)
                if value is not None:
                    values.append(str(value).strip().lower())
        joined = " ".join(values)
        if any(term in joined for term in ("constitutional", "constitution", "article", "fundamental_right", "fundamental_dut")):
            return "constitutional"
        if any(term in joined for term in ("cyber", "upi", "scam", "fraud", "online_fraud")):
            return "cyber fraud"
        if any(term in joined for term in ("criminal", "ipc", "bns", "bnss", "fir", "police", "theft", "cheating")):
            return "criminal"
        if any(term in joined for term in ("consumer", "food", "fssai", "defective", "refund")):
            return "consumer"
        if any(term in joined for term in ("property", "landlord", "tenant", "possession", "rera")):
            return "property"
        if any(term in joined for term in ("family", "divorce", "marriage", "maintenance", "custody", "hma")):
            return "family"
        if any(term in joined for term in ("document", "notice", "legal_notice", "reply", "contract", "draft")):
            return "document/legal notice"
        return "other"

    def get_admin_query_categories(self, *, period: str = "24h") -> dict[str, Any]:
        normalized_period, started_at = self._period_start(period)
        ended_at = datetime.now(timezone.utc)
        category_order = [
            "constitutional",
            "criminal",
            "consumer",
            "cyber fraud",
            "property",
            "family",
            "document/legal notice",
            "other",
        ]
        counts = {category: 0 for category in category_order}
        for message in self.chat_messages.find({"role": "assistant"}, {"metadata": 1, "created_at": 1}):
            created_at = self._parse_timestamp(message.get("created_at"))
            if created_at is None or created_at < started_at:
                continue
            metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
            category = self._category_from_metadata(metadata)
            counts[category if category in counts else "other"] += 1
        return {
            "period": normalized_period,
            "generated_at": ended_at.isoformat(),
            "categories": [{"category": category, "count": counts[category]} for category in category_order],
        }

    @staticmethod
    def _fallback_reason_from_metadata(metadata: dict[str, Any]) -> str | None:
        if not isinstance(metadata, dict):
            return None
        safe_metadata = MongoSessionStore._admin_metadata(metadata)
        fallback_reason = str(safe_metadata.get("fallback_reason") or "").strip().lower()
        validation_flags = safe_metadata.get("validation_flags")
        if isinstance(validation_flags, list):
            validation_text = " ".join(str(flag) for flag in validation_flags).lower()
        else:
            validation_text = str(validation_flags or "").lower()
        source_sufficiency = safe_metadata.get("source_sufficiency")
        if isinstance(source_sufficiency, dict):
            sufficiency_text = " ".join(str(value) for value in source_sufficiency.values()).lower()
        else:
            sufficiency_text = str(source_sufficiency or "").lower()
        combined = " ".join([fallback_reason, validation_text, sufficiency_text])
        if not combined.strip() and "fallback" not in str(safe_metadata.get("source") or "").lower():
            return None
        if any(term in combined for term in ("low_confidence", "low confidence", "retrieval_low_confidence", "confidence_low")):
            return "low confidence"
        if any(term in combined for term in ("no_relevant_authority", "no relevant authority", "no_authority")):
            return "no relevant authority"
        if any(term in combined for term in ("unsupported_output", "unsupported output")):
            return "unsupported output"
        if any(term in combined for term in ("technical_failure", "technical failure", "provider_error", "timeout")):
            return "technical failure"
        if any(term in combined for term in ("source_insufficient", "source insufficient", "insufficient", "weak")):
            return "source insufficient"
        if any(term in combined for term in ("validation_failed", "validation failed", "validator", "failed_validation")):
            return "validation failed"
        return "other"

    def get_admin_fallback_reasons(self, *, period: str = "24h") -> dict[str, Any]:
        normalized_period, started_at = self._period_start(period)
        ended_at = datetime.now(timezone.utc)
        reason_order = [
            "low confidence",
            "no relevant authority",
            "unsupported output",
            "technical failure",
            "source insufficient",
            "validation failed",
            "other",
        ]
        counts = {reason: 0 for reason in reason_order}
        for message in self.chat_messages.find({"role": "assistant"}, {"metadata": 1, "created_at": 1}):
            created_at = self._parse_timestamp(message.get("created_at"))
            if created_at is None or created_at < started_at:
                continue
            metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
            reason = self._fallback_reason_from_metadata(metadata)
            if reason is None:
                continue
            counts[reason if reason in counts else "other"] += 1
        return {
            "period": normalized_period,
            "generated_at": ended_at.isoformat(),
            "reasons": [{"reason": reason, "count": counts[reason]} for reason in reason_order],
        }

    @staticmethod
    def _percent(numerator: int | float, denominator: int | float) -> float:
        if not denominator:
            return 0.0
        return round((float(numerator) / float(denominator)) * 100, 2)

    @staticmethod
    def _quality_score(
        *,
        average_confidence: float | None,
        fallback_rate: float,
        unsupported_response_rate: float,
        thumbs_up_rate: float,
        feedback_count: int,
    ) -> int:
        confidence_component = (average_confidence * 100) if average_confidence is not None else 50.0
        feedback_component = thumbs_up_rate if feedback_count else 50.0
        raw_score = (
            confidence_component * 0.45
            + max(0.0, 100.0 - fallback_rate) * 0.25
            + max(0.0, 100.0 - unsupported_response_rate) * 0.15
            + feedback_component * 0.15
        )
        return int(round(max(0.0, min(100.0, raw_score))))

    def get_admin_quality_score(self, *, period: str = "24h") -> dict[str, Any]:
        normalized_period, started_at = self._period_start(period)
        ended_at = datetime.now(timezone.utc)
        assistant_count = 0
        fallback_count = 0
        unsupported_count = 0
        confidence_values: list[float] = []
        for message in self.chat_messages.find({"role": "assistant"}, {"metadata": 1, "created_at": 1}):
            created_at = self._parse_timestamp(message.get("created_at"))
            if created_at is None or created_at < started_at:
                continue
            assistant_count += 1
            metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
            if self._is_fallback_metadata(metadata):
                fallback_count += 1
            if self._is_unsupported_metadata(metadata):
                unsupported_count += 1
            confidence = self._numeric_confidence(metadata)
            if confidence is not None:
                confidence_values.append(confidence)

        thumbs_up = 0
        thumbs_down = 0
        for feedback in self.message_feedback.find({}, {"rating": 1, "created_at": 1}):
            created_at = self._parse_timestamp(feedback.get("created_at"))
            if created_at is None or created_at < started_at:
                continue
            rating = str(feedback.get("rating") or "").strip().lower()
            if rating == "up":
                thumbs_up += 1
            elif rating == "down":
                thumbs_down += 1

        feedback_count = thumbs_up + thumbs_down
        average_confidence = round(sum(confidence_values) / len(confidence_values), 3) if confidence_values else None
        fallback_rate = self._percent(fallback_count, assistant_count)
        unsupported_response_rate = self._percent(unsupported_count, assistant_count)
        thumbs_up_rate = self._percent(thumbs_up, feedback_count)
        thumbs_down_rate = self._percent(thumbs_down, feedback_count)
        return {
            "period": normalized_period,
            "generated_at": ended_at.isoformat(),
            "metrics": {
                "average_confidence": average_confidence,
                "fallback_rate": fallback_rate,
                "unsupported_response_rate": unsupported_response_rate,
                "thumbs_up_rate": thumbs_up_rate,
                "thumbs_down_rate": thumbs_down_rate,
                "overall_quality_score": self._quality_score(
                    average_confidence=average_confidence,
                    fallback_rate=fallback_rate,
                    unsupported_response_rate=unsupported_response_rate,
                    thumbs_up_rate=thumbs_up_rate,
                    feedback_count=feedback_count,
                ),
                "assistant_response_count": assistant_count,
                "feedback_count": feedback_count,
            },
        }

    @staticmethod
    def _low_quality_filter_matches(metadata: dict[str, Any], feedback_rating: str | None, review_filter: str) -> bool:
        normalized_filter = str(review_filter or "all").strip().lower()
        safe_metadata = MongoSessionStore._admin_metadata(metadata)
        fallback_reason = str(safe_metadata.get("fallback_reason") or "").strip().lower()
        validation_flags = safe_metadata.get("validation_flags")
        if isinstance(validation_flags, list):
            validation_text = " ".join(str(flag) for flag in validation_flags).lower()
        else:
            validation_text = str(validation_flags or "").lower()
        confidence = safe_metadata.get("confidence")
        source_sufficiency = safe_metadata.get("source_sufficiency")
        source_sufficiency_text = ""
        if isinstance(source_sufficiency, dict):
            source_sufficiency_text = str(source_sufficiency.get("label") or "").strip().lower()
        elif source_sufficiency is not None:
            source_sufficiency_text = str(source_sufficiency).strip().lower()
        is_low_confidence = False
        if isinstance(confidence, (int, float)):
            is_low_confidence = confidence <= 0.5
        else:
            is_low_confidence = "low" in str(confidence or "").strip().lower()
        is_low_confidence = is_low_confidence or source_sufficiency_text == "weak" or "low_confidence" in fallback_reason
        is_thumbs_down = str(feedback_rating or "").strip().lower() == "down"
        is_unsupported = "unsupported_output" in fallback_reason or "unsupported_output" in validation_text
        is_fallback = bool(fallback_reason) or "fallback" in str(safe_metadata.get("source") or "").strip().lower()
        if normalized_filter == "low_confidence":
            return is_low_confidence
        if normalized_filter == "thumbs_down":
            return is_thumbs_down
        if normalized_filter == "unsupported_output":
            return is_unsupported
        if normalized_filter == "fallback":
            return is_fallback
        return is_low_confidence or is_thumbs_down or is_unsupported or is_fallback

    def list_admin_low_quality_review_candidates(
        self,
        *,
        review_filter: str = "all",
        skip: int = 0,
        limit: int = 20,
    ) -> dict[str, Any]:
        normalized_filter = str(review_filter or "all").strip().lower()
        if normalized_filter not in {"all", "low_confidence", "thumbs_down", "unsupported_output", "fallback"}:
            normalized_filter = "all"
        safe_limit = max(1, min(int(limit), 100))
        safe_skip = max(0, int(skip))
        feedback_by_message: dict[int, str] = {}
        for feedback in self.message_feedback.find({}, {"message_id": 1, "rating": 1, "created_at": 1}).sort([("created_at", -1), ("id", -1)]):
            message_id = feedback.get("message_id")
            if message_id is None or message_id in feedback_by_message:
                continue
            feedback_by_message[int(message_id)] = str(feedback.get("rating") or "").strip().lower()

        matched: list[dict[str, Any]] = []
        for document in self.chat_messages.find({"role": "assistant"}).sort([("created_at", -1), ("id", -1)]):
            metadata = self._admin_metadata(document.get("metadata"))
            feedback_rating = feedback_by_message.get(int(document["id"]))
            if not self._low_quality_filter_matches(metadata, feedback_rating, normalized_filter):
                continue
            matched.append(
                {
                    "message_id": document["id"],
                    "chat_id": document["chat_id"],
                    "timestamp": document["created_at"],
                    "confidence": metadata.get("confidence"),
                    "fallback_reason": metadata.get("fallback_reason"),
                    "validation_flags": metadata.get("validation_flags") or [],
                    "feedback_rating": feedback_rating,
                }
            )
        return {
            "items": matched[safe_skip : safe_skip + safe_limit],
            "total": len(matched),
            "skip": safe_skip,
            "limit": safe_limit,
            "filter": normalized_filter,
        }

    def list_admin_users(self, *, search: str | None = None, skip: int = 0, limit: int = 20) -> dict[str, Any]:
        query: dict[str, Any] = {}
        normalized_search = (search or "").strip()
        if normalized_search:
            escaped_search = re.escape(normalized_search)
            query = {
                "$or": [
                    {"email": {"$regex": escaped_search, "$options": "i"}},
                    {"full_name": {"$regex": escaped_search, "$options": "i"}},
                ]
            }

        safe_limit = max(1, min(int(limit), 100))
        safe_skip = max(0, int(skip))
        items = []
        cursor = self.users.find(query).sort([("created_at", -1), ("id", -1)]).skip(safe_skip).limit(safe_limit)
        for document in cursor:
            public_user = self._public_user(document)
            if public_user is not None:
                items.append(public_user)
        return {
            "items": items,
            "total": self.users.count_documents(query),
            "skip": safe_skip,
            "limit": safe_limit,
        }

    @staticmethod
    def _admin_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(metadata, dict):
            return {}
        retrieval = metadata.get("retrieval")
        if not isinstance(retrieval, dict):
            retrieval = {}
        route_classification = metadata.get("route_classification")
        if not isinstance(route_classification, dict):
            route_classification = {}
        allowed_keys = {
            "route_type",
            "pipeline",
            "pipeline_path",
            "confidence",
            "fallback_reason",
            "domain",
            "source",
            "warnings",
            "validation_flags",
            "disclaimer_mode",
            "source_sufficiency",
        }
        result = {key: metadata[key] for key in allowed_keys if key in metadata}
        if "route_type" not in result:
            route_path = route_classification.get("path") or metadata.get("pipeline_path")
            if route_path:
                result["route_type"] = route_path
        if "confidence" not in result:
            confidence = retrieval.get("retrieval_confidence")
            if confidence is None:
                confidence = retrieval.get("confidence")
            if confidence is not None:
                result["confidence"] = confidence
        if "fallback_reason" not in result:
            fallback_reason = (
                retrieval.get("fallback_reason_code")
                or retrieval.get("fallback_reason")
                or retrieval.get("fallback_type")
            )
            if fallback_reason:
                result["fallback_reason"] = fallback_reason
        if "validation_flags" not in result and "validation_flags" in retrieval:
            result["validation_flags"] = retrieval.get("validation_flags")
        if "disclaimer_mode" not in result and "disclaimer_mode" in retrieval:
            result["disclaimer_mode"] = retrieval.get("disclaimer_mode")
        if "source_sufficiency" not in result and "source_sufficiency" in retrieval:
            result["source_sufficiency"] = retrieval.get("source_sufficiency")
        return result

    def list_admin_user_chats(self, user_id: int, *, skip: int = 0, limit: int = 20) -> dict[str, Any] | None:
        if self.users.find_one({"id": user_id}) is None:
            return None

        query = {"user_id": user_id}
        safe_limit = max(1, min(int(limit), 100))
        safe_skip = max(0, int(skip))
        items = []
        cursor = self.chat_sessions.find(query).sort([("updated_at", -1), ("id", -1)]).skip(safe_skip).limit(safe_limit)
        for session in cursor:
            last_message = self.chat_messages.find_one({"chat_id": session["id"]}, sort=[("id", -1)])
            last_user_message = self.chat_messages.find_one({"chat_id": session["id"], "role": "user"}, sort=[("id", -1)])
            last_metadata = self._admin_metadata((last_message or {}).get("metadata"))
            items.append(
                {
                    "id": session["id"],
                    "user_id": session.get("user_id"),
                    "title": session.get("title", "Untitled chat"),
                    "created_at": session["created_at"],
                    "updated_at": session["updated_at"],
                    "last_message_preview": (last_user_message or {}).get("content"),
                    "message_count": self.chat_messages.count_documents({"chat_id": session["id"]}),
                    "metadata": last_metadata,
                }
            )
        return {
            "items": items,
            "total": self.chat_sessions.count_documents(query),
            "skip": safe_skip,
            "limit": safe_limit,
        }

    def get_admin_chat_messages(self, chat_id: int) -> dict[str, Any] | None:
        session = self.chat_sessions.find_one({"id": chat_id})
        if session is None:
            return None

        items = []
        for document in self.chat_messages.find({"chat_id": chat_id}).sort("id", ASCENDING):
            items.append(
                {
                    "id": document["id"],
                    "chat_id": document["chat_id"],
                    "role": document["role"],
                    "content": document["content"],
                    "created_at": document["created_at"],
                    "metadata": self._admin_metadata(document.get("metadata")),
                }
            )
        return {
            "chat": {
                "id": session["id"],
                "user_id": session.get("user_id"),
                "title": session.get("title", "Untitled chat"),
                "created_at": session["created_at"],
                "updated_at": session["updated_at"],
            },
            "items": items,
            "total": len(items),
        }

    @staticmethod
    def _admin_quality_filter_matches(metadata: dict[str, Any], quality_filter: str) -> bool:
        if quality_filter == "all":
            return True
        fallback_reason = str(metadata.get("fallback_reason") or "").strip().lower()
        source = str(metadata.get("source") or "").strip().lower()
        confidence = metadata.get("confidence")
        source_sufficiency = metadata.get("source_sufficiency")
        source_sufficiency_text = ""
        if isinstance(source_sufficiency, dict):
            source_sufficiency_text = str(source_sufficiency.get("label") or "").strip().lower()
        elif source_sufficiency is not None:
            source_sufficiency_text = str(source_sufficiency).strip().lower()
        validation_flags = metadata.get("validation_flags")
        if isinstance(validation_flags, list):
            validation_text = " ".join(str(flag) for flag in validation_flags).lower()
        else:
            validation_text = str(validation_flags or "").lower()

        if quality_filter == "fallback":
            return bool(fallback_reason) or "fallback" in source
        if quality_filter == "low_confidence":
            if isinstance(confidence, (int, float)) and confidence <= 0.5:
                return True
            confidence_text = str(confidence or "").strip().lower()
            return "low" in confidence_text or source_sufficiency_text == "weak" or "low_confidence" in fallback_reason
        if quality_filter == "unsupported_output":
            return "unsupported_output" in fallback_reason or "unsupported_output" in validation_text
        return True

    def list_admin_quality_logs(self, *, quality_filter: str = "all", skip: int = 0, limit: int = 20) -> dict[str, Any]:
        normalized_filter = str(quality_filter or "all").strip().lower()
        if normalized_filter not in {"all", "fallback", "low_confidence", "unsupported_output"}:
            normalized_filter = "all"
        safe_limit = max(1, min(int(limit), 100))
        safe_skip = max(0, int(skip))
        matched: list[dict[str, Any]] = []
        cursor = self.chat_messages.find({"role": "assistant"}).sort([("created_at", -1), ("id", -1)])
        for document in cursor:
            metadata = self._admin_metadata(document.get("metadata"))
            if not self._admin_quality_filter_matches(metadata, normalized_filter):
                continue
            matched.append(
                {
                    "id": document["id"],
                    "chat_id": document["chat_id"],
                    "timestamp": document["created_at"],
                    "route_type": metadata.get("route_type"),
                    "confidence": metadata.get("confidence"),
                    "fallback_reason": metadata.get("fallback_reason"),
                    "validation_flags": metadata.get("validation_flags") or [],
                    "source_sufficiency": metadata.get("source_sufficiency"),
                }
            )
        return {
            "items": matched[safe_skip : safe_skip + safe_limit],
            "total": len(matched),
            "skip": safe_skip,
            "limit": safe_limit,
            "filter": normalized_filter,
        }

    def add_message_feedback(
        self,
        *,
        message_id: int,
        user_id: int,
        rating: str,
        comment: str | None = None,
    ) -> dict[str, Any] | None:
        message = self.chat_messages.find_one({"id": message_id, "role": "assistant"})
        if message is None:
            return None
        session = self.chat_sessions.find_one({"id": message["chat_id"], "user_id": user_id})
        if session is None:
            return None
        normalized_rating = str(rating or "").strip().lower()
        if normalized_rating not in {"up", "down"}:
            raise ValueError("Feedback rating must be up or down")
        cleaned_comment = str(comment or "").strip()
        if len(cleaned_comment) > 500:
            cleaned_comment = cleaned_comment[:500]
        now = utc_now()
        document = {
            "id": self._next_id("message_feedback"),
            "message_id": message_id,
            "chat_id": message["chat_id"],
            "user_id": user_id,
            "rating": normalized_rating,
            "comment": cleaned_comment,
            "created_at": now,
        }
        self.message_feedback.insert_one(document)
        return {
            "id": document["id"],
            "message_id": document["message_id"],
            "chat_id": document["chat_id"],
            "rating": document["rating"],
            "comment": document["comment"],
            "created_at": document["created_at"],
        }

    def list_admin_feedback(
        self,
        *,
        rating: str | None = None,
        recent_limit: int | None = None,
        skip: int = 0,
        limit: int = 20,
    ) -> dict[str, Any]:
        normalized_rating = str(rating or "").strip().lower()
        query: dict[str, Any] = {}
        if normalized_rating in {"up", "down"}:
            query["rating"] = normalized_rating
        safe_limit = max(1, min(int(limit), 100))
        safe_skip = max(0, int(skip))
        safe_recent_limit = 0
        if recent_limit is not None:
            safe_recent_limit = max(0, min(int(recent_limit), 500))
        summary = {
            "total_feedback": self.message_feedback.count_documents({}),
            "thumbs_up": self.message_feedback.count_documents({"rating": "up"}),
            "thumbs_down": self.message_feedback.count_documents({"rating": "down"}),
            "unchecked": 0,
        }
        items = []
        sorted_cursor = self.message_feedback.find(query).sort([("created_at", -1), ("id", -1)])
        matched_documents = list(sorted_cursor.limit(safe_recent_limit)) if safe_recent_limit else list(sorted_cursor)
        total = len(matched_documents)
        for document in matched_documents[safe_skip : safe_skip + safe_limit]:
            items.append(
                {
                    "id": document["id"],
                    "message_id": document["message_id"],
                    "chat_id": document["chat_id"],
                    "rating": document["rating"],
                    "comment": document.get("comment") or "",
                    "created_at": document["created_at"],
                }
            )
        return {
            "items": items,
            "total": total,
            "skip": safe_skip,
            "limit": safe_limit,
            "rating_filter": normalized_rating if normalized_rating in {"up", "down"} else "all",
            "recent_limit": safe_recent_limit,
            "summary": summary,
        }

    def record_admin_activity(
        self,
        *,
        admin_user: dict[str, Any],
        action: str,
        target_type: str | None = None,
        target_id: int | str | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        safe_action = str(action or "").strip()[:80]
        safe_target_type = str(target_type or "").strip()[:80] or None
        safe_target_id = None if target_id is None else str(target_id).strip()[:120]
        document = {
            "id": self._next_id("admin_activity"),
            "admin_user_id": admin_user.get("id"),
            "admin_email": admin_user.get("email"),
            "action": safe_action,
            "target_type": safe_target_type,
            "target_id": safe_target_id,
            "created_at": now,
        }
        self.admin_activity.insert_one(document)
        return {
            "id": document["id"],
            "admin_user_id": document["admin_user_id"],
            "admin_email": document["admin_email"],
            "action": document["action"],
            "target_type": document["target_type"],
            "target_id": document["target_id"],
            "created_at": document["created_at"],
        }

    def list_admin_activity(self, *, skip: int = 0, limit: int = 20) -> dict[str, Any]:
        safe_limit = max(1, min(int(limit), 100))
        safe_skip = max(0, int(skip))
        items = []
        cursor = self.admin_activity.find({}).sort([("created_at", -1), ("id", -1)]).skip(safe_skip).limit(safe_limit)
        for document in cursor:
            items.append(
                {
                    "id": document["id"],
                    "admin_user_id": document.get("admin_user_id"),
                    "admin_email": document.get("admin_email"),
                    "action": document.get("action"),
                    "target_type": document.get("target_type"),
                    "target_id": document.get("target_id"),
                    "created_at": document.get("created_at"),
                }
            )
        return {
            "items": items,
            "total": self.admin_activity.count_documents({}),
            "skip": safe_skip,
            "limit": safe_limit,
        }

    def get_runtime_config_bool(self, key: str) -> bool | None:
        safe_key = str(key or "").strip()
        if not safe_key:
            return None
        document = self.runtime_config.find_one({"key": safe_key})
        if document is None:
            return None
        value = document.get("value")
        return value if isinstance(value, bool) else None

    def set_runtime_config_bool(self, *, key: str, value: bool, admin_user: dict[str, Any]) -> dict[str, Any]:
        safe_key = str(key or "").strip()
        if not safe_key:
            raise ValueError("Runtime config key is required")
        now = utc_now()
        document = self.runtime_config.find_one_and_update(
            {"key": safe_key},
            {
                "$set": {
                    "key": safe_key,
                    "value": bool(value),
                    "updated_at": now,
                    "updated_by_admin_user_id": admin_user.get("id"),
                    "updated_by_admin_email": admin_user.get("email"),
                }
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return {
            "key": document["key"],
            "value": bool(document.get("value")),
            "updated_at": document.get("updated_at"),
        }

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
            self.message_feedback.delete_many({})
            self.chat_sessions.delete_many({})
            return
        session_ids = [item["id"] for item in self.chat_sessions.find({"user_id": user_id}, {"id": 1})]
        if session_ids:
            self.chat_messages.delete_many({"chat_id": {"$in": session_ids}})
            self.message_feedback.delete_many({"chat_id": {"$in": session_ids}})
        self.chat_sessions.delete_many({"user_id": user_id})

    def delete_session(self, chat_id: int, user_id: int | None = None) -> bool:
        query: dict[str, Any] = {"id": chat_id}
        if user_id is not None:
            query["user_id"] = user_id
        result = self.chat_sessions.delete_one(query)
        if result.deleted_count:
            self.chat_messages.delete_many({"chat_id": chat_id})
            self.message_feedback.delete_many({"chat_id": chat_id})
            return True
        return False
