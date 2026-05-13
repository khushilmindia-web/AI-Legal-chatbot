from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

from fastapi import Request

from backend.app.api.auth_utils import require_admin_user
from backend.app.core.config import Settings
from backend.app.api.routes import auth as auth_routes
from backend.app.services.mailer import SmtpMailer
from scripts.promote_admin import promote_admin


def test_chat_routes_require_auth(anonymous_client):
    response = anonymous_client.get("/chat/history")

    assert response.status_code == 401
    assert response.json()["detail"] == "Authentication required"


def test_signup_me_logout_and_protected_redirects(anonymous_client):
    frontend_root_response = anonymous_client.get("/frontend", follow_redirects=False)
    assert frontend_root_response.status_code == 307
    assert frontend_root_response.headers["location"] == "/frontend/auth.html"

    email = f"integration-{uuid.uuid4().hex[:8]}@example.com"
    signup_response = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Integration User",
            "email": email,
            "password": "password123",
            "state": "Gujarat",
        },
    )

    assert signup_response.status_code == 200
    assert signup_response.json()["user"]["email"] == email

    me_response = anonymous_client.get("/auth/me")
    assert me_response.status_code == 200
    assert me_response.json()["full_name"] == "Integration User"

    root_response = anonymous_client.get("/", follow_redirects=False)
    assert root_response.status_code == 307
    assert root_response.headers["location"] == "/frontend/index.html"

    legacy_index_response = anonymous_client.get("/frontend/Index.html", follow_redirects=False)
    assert legacy_index_response.status_code == 307
    assert legacy_index_response.headers["location"] == "/frontend/index.html"

    logout_response = anonymous_client.post("/auth/logout")
    assert logout_response.status_code == 200

    me_after_logout = anonymous_client.get("/auth/me")
    assert me_after_logout.status_code == 401

    protected_page = anonymous_client.get("/frontend/index.html", follow_redirects=False)
    assert protected_page.status_code == 307
    assert protected_page.headers["location"] == "/frontend/auth.html"


def test_signup_defaults_to_user_role_and_ignores_public_admin_role(anonymous_client):
    email = f"role-default-{uuid.uuid4().hex[:8]}@example.com"

    response = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Role Default User",
            "email": email,
            "password": "password123",
            "state": "Gujarat",
            "role": "admin",
            "password_hash": "should-not-be-accepted",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["user"]["role"] == "user"
    assert payload["user"]["status"] == "active"
    assert "password_hash" not in payload["user"]
    assert "auth_provider" not in payload["user"]

    stored_document = anonymous_client.app.state.session_store.users.find_one({"email": email})
    assert stored_document["role"] == "user"
    assert stored_document["status"] == "active"


def test_promote_admin_script_promotes_existing_user_without_exposing_sensitive_fields(anonymous_client):
    email = f"admin-promote-{uuid.uuid4().hex[:8]}@example.com"
    signup_response = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Admin Promote User",
            "email": email,
            "password": "password123",
            "state": "Gujarat",
        },
    )
    assert signup_response.status_code == 200
    assert signup_response.json()["user"]["role"] == "user"

    promoted = promote_admin(email)

    assert promoted["email"] == email
    assert promoted["role"] == "admin"
    assert "password_hash" not in promoted
    assert "google_sub" not in promoted

    me_response = anonymous_client.get("/auth/me")
    assert me_response.status_code == 200
    assert me_response.json()["role"] == "admin"


def install_test_admin_probe_route(app, path: str = "/__test/admin-probe") -> str:
    @app.get(path, include_in_schema=False)
    def admin_probe(request: Request):
        user = require_admin_user(request)
        return {"id": user["id"], "email": user["email"], "role": user["role"]}

    return path


def test_require_admin_user_blocks_unauthenticated_request(anonymous_client):
    path = install_test_admin_probe_route(anonymous_client.app)

    response = anonymous_client.get(path)

    assert response.status_code == 401
    assert response.json()["detail"] == "Authentication required"


def test_require_admin_user_blocks_authenticated_normal_user(client):
    path = install_test_admin_probe_route(client.app)

    response = client.get(path)

    assert response.status_code == 403
    assert response.json()["detail"] == "Admin access required"


def test_require_admin_user_allows_authenticated_admin(anonymous_client):
    path = install_test_admin_probe_route(anonymous_client.app)
    email = f"admin-auth-{uuid.uuid4().hex[:8]}@example.com"
    signup_response = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Admin Auth User",
            "email": email,
            "password": "password123",
            "state": "Gujarat",
        },
    )
    assert signup_response.status_code == 200
    promoted = promote_admin(email)
    assert promoted["role"] == "admin"

    response = anonymous_client.get(path)

    assert response.status_code == 200
    assert response.json()["email"] == email
    assert response.json()["role"] == "admin"


def test_admin_stats_blocks_unauthenticated_request(anonymous_client):
    response = anonymous_client.get("/admin/stats")

    assert response.status_code == 401
    assert response.json()["detail"] == "Authentication required"


def test_admin_me_blocks_normal_user_and_returns_safe_admin_user(client, anonymous_client):
    normal_response = client.get("/admin/me")
    assert normal_response.status_code == 403
    assert normal_response.json()["detail"] == "Admin access required"

    email = f"admin-me-{uuid.uuid4().hex[:8]}@example.com"
    signup_response = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Admin Me User",
            "email": email,
            "password": "password123",
            "state": "Gujarat",
        },
    )
    assert signup_response.status_code == 200
    assert promote_admin(email)["role"] == "admin"

    response = anonymous_client.get("/admin/me")

    assert response.status_code == 200
    payload = response.json()
    assert payload["email"] == email
    assert payload["role"] == "admin"
    assert {"password_hash", "token", "token_hash", "google_sub", "auth_provider"}.isdisjoint(payload)


def test_admin_page_requires_authenticated_session(anonymous_client):
    response = anonymous_client.get("/frontend/admin.html", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/frontend/auth.html"


def test_admin_stats_blocks_authenticated_normal_user(client):
    response = client.get("/admin/stats")

    assert response.status_code == 403
    assert response.json()["detail"] == "Admin access required"


def test_admin_stats_allows_admin_user(anonymous_client):
    email = f"admin-stats-{uuid.uuid4().hex[:8]}@example.com"
    signup_response = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Admin Stats User",
            "email": email,
            "password": "password123",
            "state": "Gujarat",
        },
    )
    assert signup_response.status_code == 200
    assert promote_admin(email)["role"] == "admin"

    response = anonymous_client.get("/admin/stats")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_users"] >= 1
    assert payload["total_chat_sessions"] >= 0
    assert payload["total_messages"] >= 0
    assert isinstance(payload["recent_sessions"], list)


def test_admin_system_status_returns_safe_operational_booleans(anonymous_client):
    email = f"admin-status-{uuid.uuid4().hex[:8]}@example.com"
    signup_response = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Admin Status User",
            "email": email,
            "password": "password123",
            "state": "Gujarat",
        },
    )
    assert signup_response.status_code == 200
    assert promote_admin(email)["role"] == "admin"

    settings = anonymous_client.app.state.settings
    settings.indiankanoon_api_token = "ik-secret-token"
    settings.google_search_enabled = True
    settings.google_custom_search_api_key = "google-secret-key"
    settings.google_custom_search_cx = "google-secret-cx"
    settings.openai_api_key = "openai-secret-key"
    settings.smtp_host = "smtp.example.com"
    settings.smtp_username = "sender@example.com"
    settings.smtp_password = "smtp-secret-password"
    settings.smtp_from_email = "sender@example.com"

    response = anonymous_client.get("/admin/system/status")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "mongodb_reachable",
        "indiankanoon_configured",
        "google_custom_search_configured",
        "openai_configured",
        "smtp_configured",
        "services",
    }
    assert payload["mongodb_reachable"] is True
    assert payload["indiankanoon_configured"] is True
    assert payload["google_custom_search_configured"] is True
    assert payload["openai_configured"] is True
    assert payload["smtp_configured"] is True
    assert payload["services"]["mongodb"]["status"] in {"online", "offline"}
    assert payload["services"]["indiankanoon"]["status"] == "configured"
    assert payload["services"]["google_custom_search"]["status"] == "configured"
    assert payload["services"]["openai"]["status"] == "configured"
    assert payload["services"]["smtp"]["status"] == "configured"

    serialized = json.dumps(payload).lower()
    forbidden_values = [
        "ik-secret-token",
        "google-secret-key",
        "google-secret-cx",
        "openai-secret-key",
        "smtp-secret-password",
        "mongodb://",
        "sender@example.com",
    ]
    for forbidden in forbidden_values:
        assert forbidden not in serialized


def test_admin_system_health_returns_safe_metrics_and_handles_degraded_database(anonymous_client):
    email = f"admin-health-{uuid.uuid4().hex[:8]}@example.com"
    signup_response = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Admin Health User",
            "email": email,
            "password": "password123",
            "state": "Gujarat",
        },
    )
    assert signup_response.status_code == 200
    assert promote_admin(email)["role"] == "admin"
    anonymous_client.app.state.settings.openai_api_key = "openai-secret-key"
    anonymous_client.app.state.health_metrics["chat_response_times_ms"] = [125.0, 175.0]

    response = anonymous_client.get("/admin/system/health")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "status",
        "statuses",
        "checked_at",
        "uptime",
        "database",
        "ai_provider",
        "chat",
        "sessions",
        "memory",
        "cpu",
    }
    assert payload["status"] in {"healthy", "warning", "critical"}
    assert payload["statuses"] == ["critical", "healthy", "warning"]
    assert payload["uptime"]["seconds"] >= 0
    assert payload["database"]["reachable"] is True
    assert payload["database"]["response_time_ms"] is not None
    assert payload["ai_provider"]["providers"]["openai"] == "configured"
    assert payload["chat"]["average_response_time_ms"] == 150.0
    assert payload["chat"]["sample_size"] == 2
    assert payload["sessions"]["active"] >= 1
    assert payload["memory"]["status"] in {"healthy", "warning", "critical"}
    assert payload["cpu"]["status"] in {"healthy", "warning", "critical"}
    serialized = json.dumps(payload).lower()
    for forbidden in [
        "openai-secret-key",
        "password_hash",
        "token",
        "mongodb://",
        "d:\\",
        "session_token",
        "api_key",
        "secret",
    ]:
        assert forbidden not in serialized

    original_client = anonymous_client.app.state.session_store.client
    anonymous_client.app.state.session_store.client = None
    try:
        degraded_response = anonymous_client.get("/admin/system/health")
    finally:
        anonymous_client.app.state.session_store.client = original_client

    assert degraded_response.status_code == 200
    degraded_payload = degraded_response.json()
    assert degraded_payload["status"] == "critical"
    assert degraded_payload["database"]["status"] == "critical"
    assert degraded_payload["database"]["reachable"] is False

    activity = anonymous_client.app.state.session_store.admin_activity.find_one({"action": "viewed_system_health"})
    assert activity is not None


def test_admin_analytics_overview_returns_safe_aggregates_by_period(anonymous_client):
    admin_email = f"admin-analytics-{uuid.uuid4().hex[:8]}@example.com"
    user_email = f"analytics-user-{uuid.uuid4().hex[:8]}@example.com"
    old_user_email = f"analytics-old-{uuid.uuid4().hex[:8]}@example.com"
    admin_signup = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Admin Analytics User",
            "email": admin_email,
            "password": "password123",
            "state": "Gujarat",
        },
    )
    assert admin_signup.status_code == 200
    assert promote_admin(admin_email)["role"] == "admin"
    anonymous_client.post("/auth/logout")
    user_signup = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Analytics Normal User",
            "email": user_email,
            "password": "password123",
            "state": "Delhi",
        },
    )
    assert user_signup.status_code == 200
    user_id = user_signup.json()["user"]["id"]
    anonymous_client.post("/auth/logout")
    old_user_signup = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Analytics Old User",
            "email": old_user_email,
            "password": "password123",
            "state": "Maharashtra",
        },
    )
    assert old_user_signup.status_code == 200
    old_user_id = old_user_signup.json()["user"]["id"]

    store = anonymous_client.app.state.session_store
    recent_chat = store.create_session("Recent analytics chat", user_id=user_id)
    store.add_message(recent_chat["id"], "user", "Sensitive raw user question", metadata={}, user_id=user_id)
    store.add_message(
        recent_chat["id"],
        "assistant",
        "Safe assistant answer",
        metadata={"confidence": 0.8, "fallback_reason": "retrieval_low_confidence"},
        user_id=user_id,
    )
    store.add_message(
        recent_chat["id"],
        "assistant",
        "Unsupported fallback answer",
        metadata={"confidence": 0.4, "fallback_reason": "unsupported_output", "token": "must-not-leak"},
        user_id=user_id,
    )

    older_chat = store.create_session("Old analytics chat", user_id=old_user_id)
    old_timestamp = (datetime.now(timezone.utc) - timedelta(days=9)).isoformat()
    store.chat_sessions.update_one(
        {"id": older_chat["id"]},
        {"$set": {"created_at": old_timestamp, "updated_at": old_timestamp}},
    )
    store.add_message(
        older_chat["id"],
        "assistant",
        "Older assistant answer",
        metadata={"confidence": 1.0},
        user_id=old_user_id,
    )
    store.chat_sessions.update_one(
        {"id": older_chat["id"]},
        {"$set": {"created_at": old_timestamp, "updated_at": old_timestamp}},
    )
    store.chat_messages.update_one({"chat_id": older_chat["id"]}, {"$set": {"created_at": old_timestamp}})

    anonymous_client.app.state.health_metrics["chat_response_times_ms"] = [100.0, 200.0]
    anonymous_client.post("/auth/logout")
    assert anonymous_client.post("/auth/login", json={"email": admin_email, "password": "password123"}).status_code == 200

    response = anonymous_client.get("/admin/analytics/overview", params={"period": "24h"})

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"period", "generated_at", "window", "metrics"}
    assert payload["period"] == "24h"
    assert set(payload["metrics"]) == {
        "total_chats",
        "total_messages",
        "active_users",
        "fallback_count",
        "unsupported_response_count",
        "average_confidence",
        "average_response_time_ms",
    }
    assert payload["metrics"]["total_chats"] == 1
    assert payload["metrics"]["total_messages"] == 3
    assert payload["metrics"]["active_users"] == 1
    assert payload["metrics"]["fallback_count"] == 2
    assert payload["metrics"]["unsupported_response_count"] == 1
    assert payload["metrics"]["average_confidence"] == 0.6
    assert payload["metrics"]["average_response_time_ms"] == 150.0
    serialized = json.dumps(payload).lower()
    for forbidden in [
        "sensitive raw user question",
        user_email,
        old_user_email,
        "safe assistant answer",
        "must-not-leak",
        "password_hash",
        "token",
        "google_sub",
        "auth_provider",
    ]:
        assert forbidden not in serialized

    seven_day_payload = anonymous_client.get("/admin/analytics/overview", params={"period": "7d"}).json()
    thirty_day_payload = anonymous_client.get("/admin/analytics/overview", params={"period": "30d"}).json()
    assert seven_day_payload["metrics"]["total_chats"] == 1
    assert thirty_day_payload["metrics"]["total_chats"] == 2

    invalid_response = anonymous_client.get("/admin/analytics/overview", params={"period": "90d"})
    assert invalid_response.status_code == 422
    assert store.admin_activity.find_one({"action": "viewed_analytics_overview", "target_id": "24h"}) is not None


def test_admin_analytics_trends_returns_safe_daily_aggregates(anonymous_client):
    admin_email = f"admin-trends-{uuid.uuid4().hex[:8]}@example.com"
    user_email = f"trends-user-{uuid.uuid4().hex[:8]}@example.com"
    old_user_email = f"trends-old-{uuid.uuid4().hex[:8]}@example.com"
    assert anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Admin Trends User",
            "email": admin_email,
            "password": "password123",
            "state": "Gujarat",
        },
    ).status_code == 200
    assert promote_admin(admin_email)["role"] == "admin"
    anonymous_client.post("/auth/logout")
    user_signup = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Trends Normal User",
            "email": user_email,
            "password": "password123",
            "state": "Delhi",
        },
    )
    assert user_signup.status_code == 200
    user_id = user_signup.json()["user"]["id"]
    anonymous_client.post("/auth/logout")
    old_signup = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Trends Old User",
            "email": old_user_email,
            "password": "password123",
            "state": "Maharashtra",
        },
    )
    assert old_signup.status_code == 200
    old_user_id = old_signup.json()["user"]["id"]
    store = anonymous_client.app.state.session_store
    today = datetime.now(timezone.utc)
    yesterday = today - timedelta(days=1)
    older = today - timedelta(days=10)

    today_chat = store.create_session("Today trends chat", user_id=user_id)
    store.add_message(today_chat["id"], "user", "private today message", metadata={}, user_id=user_id)
    store.add_message(
        today_chat["id"],
        "assistant",
        "today assistant",
        metadata={"confidence": 0.8, "fallback_reason": "retrieval_low_confidence"},
        user_id=user_id,
    )
    store.chat_sessions.update_one(
        {"id": today_chat["id"]},
        {"$set": {"created_at": today.isoformat(), "updated_at": today.isoformat()}},
    )
    store.chat_messages.update_many({"chat_id": today_chat["id"]}, {"$set": {"created_at": today.isoformat()}})

    yesterday_chat = store.create_session("Yesterday trends chat", user_id=user_id)
    store.add_message(
        yesterday_chat["id"],
        "assistant",
        "yesterday assistant",
        metadata={"confidence": 0.4},
        user_id=user_id,
    )
    store.add_message(
        yesterday_chat["id"],
        "assistant",
        "yesterday fallback",
        metadata={"confidence": 0.6, "fallback_reason": "technical_failure"},
        user_id=user_id,
    )
    store.chat_sessions.update_one(
        {"id": yesterday_chat["id"]},
        {"$set": {"created_at": yesterday.isoformat(), "updated_at": yesterday.isoformat()}},
    )
    store.chat_messages.update_many({"chat_id": yesterday_chat["id"]}, {"$set": {"created_at": yesterday.isoformat()}})

    older_chat = store.create_session("Older trends chat", user_id=old_user_id)
    store.add_message(older_chat["id"], "assistant", "older assistant", metadata={"confidence": 1.0}, user_id=old_user_id)
    store.chat_sessions.update_one(
        {"id": older_chat["id"]},
        {"$set": {"created_at": older.isoformat(), "updated_at": older.isoformat()}},
    )
    store.chat_messages.update_many({"chat_id": older_chat["id"]}, {"$set": {"created_at": older.isoformat()}})

    anonymous_client.post("/auth/logout")
    assert anonymous_client.post("/auth/login", json={"email": admin_email, "password": "password123"}).status_code == 200

    response = anonymous_client.get("/admin/analytics/trends", params={"period": "7d"})

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"period", "generated_at", "items"}
    assert payload["period"] == "7d"
    assert len(payload["items"]) == 7
    for item in payload["items"]:
        assert set(item) == {"date", "chats", "messages", "fallback_count", "average_confidence"}
    by_date = {item["date"]: item for item in payload["items"]}
    today_item = by_date[today.date().isoformat()]
    yesterday_item = by_date[yesterday.date().isoformat()]
    assert today_item["chats"] == 1
    assert today_item["messages"] == 2
    assert today_item["fallback_count"] == 1
    assert today_item["average_confidence"] == 0.8
    assert yesterday_item["chats"] == 1
    assert yesterday_item["messages"] == 2
    assert yesterday_item["fallback_count"] == 1
    assert yesterday_item["average_confidence"] == 0.5

    thirty_day_payload = anonymous_client.get("/admin/analytics/trends", params={"period": "30d"}).json()
    assert len(thirty_day_payload["items"]) == 30
    assert thirty_day_payload["items"][0]["date"] <= older.date().isoformat()
    assert {item["date"] for item in thirty_day_payload["items"]} >= {older.date().isoformat()}

    invalid_response = anonymous_client.get("/admin/analytics/trends", params={"period": "24h"})
    assert invalid_response.status_code == 422
    assert store.admin_activity.find_one({"action": "viewed_analytics_trends", "target_id": "7d"}) is not None

    serialized = json.dumps(payload).lower()
    for forbidden in [
        "private today message",
        "today assistant",
        "yesterday assistant",
        user_email,
        old_user_email,
        "password_hash",
        "token",
        "google_sub",
        "auth_provider",
    ]:
        assert forbidden not in serialized


def test_admin_query_categories_returns_safe_aggregate_counts(anonymous_client):
    admin_email = f"admin-categories-{uuid.uuid4().hex[:8]}@example.com"
    user_email = f"categories-user-{uuid.uuid4().hex[:8]}@example.com"
    old_user_email = f"categories-old-{uuid.uuid4().hex[:8]}@example.com"
    assert anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Admin Categories User",
            "email": admin_email,
            "password": "password123",
            "state": "Gujarat",
        },
    ).status_code == 200
    assert promote_admin(admin_email)["role"] == "admin"
    anonymous_client.post("/auth/logout")
    user_signup = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Categories Normal User",
            "email": user_email,
            "password": "password123",
            "state": "Delhi",
        },
    )
    assert user_signup.status_code == 200
    user_id = user_signup.json()["user"]["id"]
    anonymous_client.post("/auth/logout")
    old_signup = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Categories Old User",
            "email": old_user_email,
            "password": "password123",
            "state": "Maharashtra",
        },
    )
    assert old_signup.status_code == 200
    old_user_id = old_signup.json()["user"]["id"]
    store = anonymous_client.app.state.session_store
    metadata_by_category = [
        {"domain": "constitutional"},
        {"domain": "criminal"},
        {"domain": "consumer"},
        {"domain": "cyber fraud"},
        {"domain": "property"},
        {"domain": "family"},
        {"route_type": "document_review", "playbook_id": "legal_notice_reply"},
        {"domain": "unknown-topic"},
    ]
    for index, metadata in enumerate(metadata_by_category):
        session = store.create_session(f"Category chat {index}", user_id=user_id)
        store.add_message(
            session["id"],
            "assistant",
            f"private category answer {index}",
            metadata={**metadata, "email": "must-not-leak@example.com", "token": "must-not-leak"},
            user_id=user_id,
        )

    old_session = store.create_session("Old category chat", user_id=old_user_id)
    old_timestamp = (datetime.now(timezone.utc) - timedelta(days=9)).isoformat()
    store.add_message(old_session["id"], "assistant", "old constitutional", metadata={"domain": "constitutional"}, user_id=old_user_id)
    store.chat_sessions.update_one(
        {"id": old_session["id"]},
        {"$set": {"created_at": old_timestamp, "updated_at": old_timestamp}},
    )
    store.chat_messages.update_many({"chat_id": old_session["id"]}, {"$set": {"created_at": old_timestamp}})

    anonymous_client.post("/auth/logout")
    assert anonymous_client.post("/auth/login", json={"email": admin_email, "password": "password123"}).status_code == 200

    response = anonymous_client.get("/admin/analytics/query-categories", params={"period": "24h"})

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"period", "generated_at", "categories"}
    assert payload["period"] == "24h"
    counts = {item["category"]: item["count"] for item in payload["categories"]}
    assert counts == {
        "constitutional": 1,
        "criminal": 1,
        "consumer": 1,
        "cyber fraud": 1,
        "property": 1,
        "family": 1,
        "document/legal notice": 1,
        "other": 1,
    }

    seven_day_counts = {
        item["category"]: item["count"]
        for item in anonymous_client.get("/admin/analytics/query-categories", params={"period": "7d"}).json()["categories"]
    }
    thirty_day_counts = {
        item["category"]: item["count"]
        for item in anonymous_client.get("/admin/analytics/query-categories", params={"period": "30d"}).json()["categories"]
    }
    assert seven_day_counts["constitutional"] == 1
    assert thirty_day_counts["constitutional"] == 2
    invalid_response = anonymous_client.get("/admin/analytics/query-categories", params={"period": "90d"})
    assert invalid_response.status_code == 422
    assert store.admin_activity.find_one({"action": "viewed_query_categories", "target_id": "24h"}) is not None

    serialized = json.dumps(payload).lower()
    for forbidden in [
        "private category answer",
        user_email,
        old_user_email,
        "must-not-leak",
        "password_hash",
        "token",
        "google_sub",
        "auth_provider",
        "email",
    ]:
        assert forbidden not in serialized


def test_admin_fallback_reasons_returns_safe_aggregate_counts(anonymous_client):
    admin_email = f"admin-fallbacks-{uuid.uuid4().hex[:8]}@example.com"
    user_email = f"fallbacks-user-{uuid.uuid4().hex[:8]}@example.com"
    old_user_email = f"fallbacks-old-{uuid.uuid4().hex[:8]}@example.com"
    assert anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Admin Fallback User",
            "email": admin_email,
            "password": "password123",
            "state": "Gujarat",
        },
    ).status_code == 200
    assert promote_admin(admin_email)["role"] == "admin"
    anonymous_client.post("/auth/logout")
    user_signup = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Fallback Normal User",
            "email": user_email,
            "password": "password123",
            "state": "Delhi",
        },
    )
    assert user_signup.status_code == 200
    user_id = user_signup.json()["user"]["id"]
    anonymous_client.post("/auth/logout")
    old_signup = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Fallback Old User",
            "email": old_user_email,
            "password": "password123",
            "state": "Maharashtra",
        },
    )
    assert old_signup.status_code == 200
    old_user_id = old_signup.json()["user"]["id"]
    store = anonymous_client.app.state.session_store
    metadata_by_reason = [
        {"fallback_reason": "retrieval_low_confidence"},
        {"retrieval": {"fallback_reason_code": "no_relevant_authority"}},
        {"fallback_reason": "unsupported_output"},
        {"fallback_reason": "technical_failure"},
        {"source_sufficiency": {"label": "weak", "reason": "source_insufficient"}},
        {"validation_flags": ["validation_failed"]},
        {"source": "deterministic_fallback"},
    ]
    for index, metadata in enumerate(metadata_by_reason):
        session = store.create_session(f"Fallback reason chat {index}", user_id=user_id)
        store.add_message(
            session["id"],
            "assistant",
            f"private fallback answer {index}",
            metadata={**metadata, "email": "must-not-leak@example.com", "token": "must-not-leak"},
            user_id=user_id,
        )

    non_fallback_session = store.create_session("Normal answer chat", user_id=user_id)
    store.add_message(
        non_fallback_session["id"],
        "assistant",
        "normal assistant answer",
        metadata={"confidence": 0.95, "domain": "constitutional"},
        user_id=user_id,
    )

    old_session = store.create_session("Old fallback chat", user_id=old_user_id)
    old_timestamp = (datetime.now(timezone.utc) - timedelta(days=9)).isoformat()
    store.add_message(old_session["id"], "assistant", "old fallback", metadata={"fallback_reason": "technical_failure"}, user_id=old_user_id)
    store.chat_sessions.update_one(
        {"id": old_session["id"]},
        {"$set": {"created_at": old_timestamp, "updated_at": old_timestamp}},
    )
    store.chat_messages.update_many({"chat_id": old_session["id"]}, {"$set": {"created_at": old_timestamp}})

    anonymous_client.post("/auth/logout")
    assert anonymous_client.post("/auth/login", json={"email": admin_email, "password": "password123"}).status_code == 200

    response = anonymous_client.get("/admin/analytics/fallback-reasons", params={"period": "24h"})

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"period", "generated_at", "reasons"}
    assert payload["period"] == "24h"
    counts = {item["reason"]: item["count"] for item in payload["reasons"]}
    assert counts == {
        "low confidence": 1,
        "no relevant authority": 1,
        "unsupported output": 1,
        "technical failure": 1,
        "source insufficient": 1,
        "validation failed": 1,
        "other": 1,
    }
    seven_day_counts = {
        item["reason"]: item["count"]
        for item in anonymous_client.get("/admin/analytics/fallback-reasons", params={"period": "7d"}).json()["reasons"]
    }
    thirty_day_counts = {
        item["reason"]: item["count"]
        for item in anonymous_client.get("/admin/analytics/fallback-reasons", params={"period": "30d"}).json()["reasons"]
    }
    assert seven_day_counts["technical failure"] == 1
    assert thirty_day_counts["technical failure"] == 2
    invalid_response = anonymous_client.get("/admin/analytics/fallback-reasons", params={"period": "90d"})
    assert invalid_response.status_code == 422
    assert store.admin_activity.find_one({"action": "viewed_fallback_reasons", "target_id": "24h"}) is not None

    serialized = json.dumps(payload).lower()
    for forbidden in [
        "private fallback answer",
        "normal assistant answer",
        user_email,
        old_user_email,
        "must-not-leak",
        "password_hash",
        "token",
        "google_sub",
        "auth_provider",
        "email",
    ]:
        assert forbidden not in serialized


def test_admin_quality_score_returns_safe_aggregate_metrics(anonymous_client):
    admin_email = f"admin-quality-score-{uuid.uuid4().hex[:8]}@example.com"
    user_email = f"quality-score-user-{uuid.uuid4().hex[:8]}@example.com"
    old_user_email = f"quality-score-old-{uuid.uuid4().hex[:8]}@example.com"
    assert anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Admin Quality Score User",
            "email": admin_email,
            "password": "password123",
            "state": "Gujarat",
        },
    ).status_code == 200
    assert promote_admin(admin_email)["role"] == "admin"
    anonymous_client.post("/auth/logout")
    user_signup = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Quality Score Normal User",
            "email": user_email,
            "password": "password123",
            "state": "Delhi",
        },
    )
    assert user_signup.status_code == 200
    user_id = user_signup.json()["user"]["id"]
    anonymous_client.post("/auth/logout")
    old_signup = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Quality Score Old User",
            "email": old_user_email,
            "password": "password123",
            "state": "Maharashtra",
        },
    )
    assert old_signup.status_code == 200
    old_user_id = old_signup.json()["user"]["id"]
    store = anonymous_client.app.state.session_store
    assistant_metadata = [
        {"confidence": 0.9},
        {"confidence": 0.7, "fallback_reason": "retrieval_low_confidence"},
        {"confidence": 0.5, "fallback_reason": "unsupported_output", "validation_flags": ["unsupported_output"]},
        {"retrieval": {"retrieval_confidence": 0.3, "fallback_reason_code": "technical_failure"}},
    ]
    assistant_ids = []
    for index, metadata in enumerate(assistant_metadata):
        session = store.create_session(f"Quality score chat {index}", user_id=user_id)
        message = store.add_message(
            session["id"],
            "assistant",
            f"private quality answer {index}",
            metadata={**metadata, "email": "must-not-leak@example.com", "token": "must-not-leak"},
            user_id=user_id,
        )
        assistant_ids.append(message["id"])

    store.add_message_feedback(message_id=assistant_ids[0], user_id=user_id, rating="up", comment="private positive")
    store.add_message_feedback(message_id=assistant_ids[1], user_id=user_id, rating="down", comment="private negative")

    old_session = store.create_session("Old quality score chat", user_id=old_user_id)
    old_timestamp = (datetime.now(timezone.utc) - timedelta(days=9)).isoformat()
    old_message = store.add_message(old_session["id"], "assistant", "old quality answer", metadata={"confidence": 1.0}, user_id=old_user_id)
    old_feedback = store.add_message_feedback(message_id=old_message["id"], user_id=old_user_id, rating="up", comment="old private")
    store.chat_sessions.update_one(
        {"id": old_session["id"]},
        {"$set": {"created_at": old_timestamp, "updated_at": old_timestamp}},
    )
    store.chat_messages.update_many({"chat_id": old_session["id"]}, {"$set": {"created_at": old_timestamp}})
    store.message_feedback.update_one({"id": old_feedback["id"]}, {"$set": {"created_at": old_timestamp}})

    anonymous_client.post("/auth/logout")
    assert anonymous_client.post("/auth/login", json={"email": admin_email, "password": "password123"}).status_code == 200

    response = anonymous_client.get("/admin/analytics/quality-score", params={"period": "24h"})

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"period", "generated_at", "metrics"}
    assert payload["period"] == "24h"
    assert payload["metrics"] == {
        "average_confidence": 0.6,
        "fallback_rate": 75.0,
        "unsupported_response_rate": 25.0,
        "thumbs_up_rate": 50.0,
        "thumbs_down_rate": 50.0,
        "overall_quality_score": 52,
        "assistant_response_count": 4,
        "feedback_count": 2,
    }

    seven_day_payload = anonymous_client.get("/admin/analytics/quality-score", params={"period": "7d"}).json()
    thirty_day_payload = anonymous_client.get("/admin/analytics/quality-score", params={"period": "30d"}).json()
    assert seven_day_payload["metrics"]["assistant_response_count"] == 4
    assert thirty_day_payload["metrics"]["assistant_response_count"] == 5
    assert thirty_day_payload["metrics"]["feedback_count"] == 3
    invalid_response = anonymous_client.get("/admin/analytics/quality-score", params={"period": "90d"})
    assert invalid_response.status_code == 422
    assert store.admin_activity.find_one({"action": "viewed_quality_score", "target_id": "24h"}) is not None

    serialized = json.dumps(payload).lower()
    for forbidden in [
        "private quality answer",
        "private positive",
        "private negative",
        user_email,
        old_user_email,
        "must-not-leak",
        "password_hash",
        "token",
        "google_sub",
        "auth_provider",
        "email",
    ]:
        assert forbidden not in serialized


def test_admin_low_quality_review_queue_returns_safe_candidates_and_filters(anonymous_client):
    admin_email = f"admin-review-{uuid.uuid4().hex[:8]}@example.com"
    user_email = f"review-user-{uuid.uuid4().hex[:8]}@example.com"
    assert anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Admin Review User",
            "email": admin_email,
            "password": "password123",
            "state": "Gujarat",
        },
    ).status_code == 200
    assert promote_admin(admin_email)["role"] == "admin"
    anonymous_client.post("/auth/logout")
    user_signup = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Review Normal User",
            "email": user_email,
            "password": "password123",
            "state": "Delhi",
        },
    )
    assert user_signup.status_code == 200
    user_id = user_signup.json()["user"]["id"]
    store = anonymous_client.app.state.session_store
    session = store.create_session("Review queue chat", user_id=user_id)
    low_confidence = store.add_message(
        session["id"],
        "assistant",
        "private low confidence answer",
        metadata={"confidence": 0.4, "email": "must-not-leak@example.com", "token": "must-not-leak"},
        user_id=user_id,
    )
    fallback = store.add_message(
        session["id"],
        "assistant",
        "private fallback answer",
        metadata={"confidence": 0.8, "fallback_reason": "technical_failure"},
        user_id=user_id,
    )
    unsupported = store.add_message(
        session["id"],
        "assistant",
        "private unsupported answer",
        metadata={"confidence": 0.7, "fallback_reason": "unsupported_output", "validation_flags": ["unsupported_output"]},
        user_id=user_id,
    )
    thumbs_down = store.add_message(
        session["id"],
        "assistant",
        "private thumbs down answer",
        metadata={"confidence": 0.9},
        user_id=user_id,
    )
    good = store.add_message(
        session["id"],
        "assistant",
        "private good answer",
        metadata={"confidence": 0.95},
        user_id=user_id,
    )
    store.add_message_feedback(message_id=thumbs_down["id"], user_id=user_id, rating="down", comment="private bad")
    store.add_message_feedback(message_id=good["id"], user_id=user_id, rating="up", comment="private good")

    anonymous_client.post("/auth/logout")
    assert anonymous_client.post("/auth/login", json={"email": admin_email, "password": "password123"}).status_code == 200

    response = anonymous_client.get("/admin/review/low-quality", params={"review_filter": "all", "skip": 0, "limit": 10})

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"items", "total", "skip", "limit", "filter"}
    assert payload["total"] == 4
    assert payload["filter"] == "all"
    candidate_ids = {item["message_id"] for item in payload["items"]}
    assert candidate_ids == {low_confidence["id"], fallback["id"], unsupported["id"], thumbs_down["id"]}
    for item in payload["items"]:
        assert set(item) == {
            "message_id",
            "chat_id",
            "timestamp",
            "confidence",
            "fallback_reason",
            "validation_flags",
            "feedback_rating",
        }
        assert item["chat_id"] == session["id"]

    filter_expectations = {
        "low_confidence": {low_confidence["id"]},
        "thumbs_down": {thumbs_down["id"]},
        "unsupported_output": {unsupported["id"]},
        "fallback": {fallback["id"], unsupported["id"]},
    }
    for review_filter, expected_ids in filter_expectations.items():
        filtered = anonymous_client.get("/admin/review/low-quality", params={"review_filter": review_filter}).json()
        assert {item["message_id"] for item in filtered["items"]} == expected_ids

    invalid_response = anonymous_client.get("/admin/review/low-quality", params={"review_filter": "delete"})
    assert invalid_response.status_code == 422
    assert store.admin_activity.find_one({"action": "viewed_low_quality_review", "target_id": "all"}) is not None
    serialized = json.dumps(payload).lower()
    for forbidden in [
        "private low confidence answer",
        "private fallback answer",
        "private unsupported answer",
        "private thumbs down answer",
        "private bad",
        user_email,
        "must-not-leak",
        "password_hash",
        "token",
        "google_sub",
        "auth_provider",
        "email",
    ]:
        assert forbidden not in serialized


def test_admin_config_returns_safe_read_only_feature_toggles(anonymous_client):
    email = f"admin-config-{uuid.uuid4().hex[:8]}@example.com"
    signup_response = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Admin Config User",
            "email": email,
            "password": "password123",
            "state": "Gujarat",
        },
    )
    assert signup_response.status_code == 200
    assert promote_admin(email)["role"] == "admin"

    settings = anonymous_client.app.state.settings
    settings.indiankanoon_api_token = "ik-secret-token"
    settings.google_search_enabled = True
    settings.google_custom_search_api_key = "google-secret-key"
    settings.google_custom_search_cx = "google-secret-cx"
    settings.openai_api_key = "openai-secret-key"
    settings.smtp_password = "smtp-secret-password"
    settings.maintenance_mode = True

    response = anonymous_client.get("/admin/config")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"toggles", "enabled_count", "total_count"}
    assert set(payload["toggles"]) == {
        "enable_google_fallback",
        "enable_india_kanoon",
        "enable_direct_answers",
        "enable_feedback",
        "maintenance_mode",
    }
    assert payload["toggles"]["enable_google_fallback"]["enabled"] is True
    assert payload["toggles"]["enable_india_kanoon"]["enabled"] is True
    assert payload["toggles"]["enable_direct_answers"]["enabled"] is True
    assert payload["toggles"]["enable_feedback"]["enabled"] is True
    assert payload["toggles"]["maintenance_mode"]["enabled"] is True
    assert payload["total_count"] == 5
    assert payload["enabled_count"] == 5
    for toggle in payload["toggles"].values():
        assert set(toggle) == {"label", "enabled", "status"}
        assert toggle["status"] in {"enabled", "disabled"}

    serialized = json.dumps(payload).lower()
    for forbidden in [
        "ik-secret-token",
        "google-secret-key",
        "google-secret-cx",
        "openai-secret-key",
        "smtp-secret-password",
        "api_key",
        "token",
        "secret",
        "password",
        "mongodb://",
    ]:
        assert forbidden not in serialized


def test_maintenance_mode_blocks_normal_user_from_chat(anonymous_client):
    email = f"maintenance-user-{uuid.uuid4().hex[:8]}@example.com"
    signup_response = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Maintenance User",
            "email": email,
            "password": "password123",
            "state": "Gujarat",
        },
    )
    assert signup_response.status_code == 200
    anonymous_client.app.state.settings.maintenance_mode = True

    response = anonymous_client.post("/chat", json={"message": "What is Article 21?"})

    assert response.status_code == 503
    assert response.json()["detail"] == "The system is temporarily under maintenance. Please try again later."


def test_maintenance_mode_allows_admin_user_to_chat(anonymous_client):
    email = f"maintenance-admin-{uuid.uuid4().hex[:8]}@example.com"
    signup_response = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Maintenance Admin",
            "email": email,
            "password": "password123",
            "state": "Gujarat",
        },
    )
    assert signup_response.status_code == 200
    assert promote_admin(email)["role"] == "admin"
    anonymous_client.app.state.settings.maintenance_mode = True

    response = anonymous_client.post("/chat", json={"message": "What is Article 21?"})

    assert response.status_code == 200
    assert "answer" in response.json()


def test_maintenance_mode_keeps_auth_and_admin_endpoints_available(anonymous_client):
    normal_email = f"maintenance-auth-{uuid.uuid4().hex[:8]}@example.com"
    signup_response = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Maintenance Auth User",
            "email": normal_email,
            "password": "password123",
            "state": "Gujarat",
        },
    )
    assert signup_response.status_code == 200
    anonymous_client.app.state.settings.maintenance_mode = True

    me_response = anonymous_client.get("/auth/me")
    logout_response = anonymous_client.post("/auth/logout")
    login_response = anonymous_client.post(
        "/auth/login",
        json={"email": normal_email, "password": "password123"},
    )

    assert me_response.status_code == 200
    assert logout_response.status_code == 200
    assert login_response.status_code == 200

    admin_email = f"maintenance-admin-endpoint-{uuid.uuid4().hex[:8]}@example.com"
    admin_signup_response = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Maintenance Endpoint Admin",
            "email": admin_email,
            "password": "password123",
            "state": "Gujarat",
        },
    )
    assert admin_signup_response.status_code == 200
    assert promote_admin(admin_email)["role"] == "admin"

    admin_response = anonymous_client.get("/admin/config")

    assert admin_response.status_code == 200
    assert admin_response.json()["toggles"]["maintenance_mode"]["enabled"] is True


def test_admin_can_update_maintenance_mode_in_runtime_config_only(anonymous_client):
    admin_email = f"runtime-maintenance-admin-{uuid.uuid4().hex[:8]}@example.com"
    user_email = f"runtime-maintenance-user-{uuid.uuid4().hex[:8]}@example.com"
    admin_signup = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Runtime Maintenance Admin",
            "email": admin_email,
            "password": "password123",
            "state": "Gujarat",
        },
    )
    assert admin_signup.status_code == 200
    assert promote_admin(admin_email)["role"] == "admin"
    anonymous_client.app.state.settings.maintenance_mode = False

    enable_response = anonymous_client.patch("/admin/config/maintenance", json={"maintenance_mode": True})

    assert enable_response.status_code == 200
    assert enable_response.json()["toggles"]["maintenance_mode"]["enabled"] is True
    assert anonymous_client.app.state.settings.maintenance_mode is False
    runtime_document = anonymous_client.app.state.session_store.runtime_config.find_one({"key": "maintenance_mode"})
    assert runtime_document["value"] is True
    assert runtime_document["updated_by_admin_email"] == admin_email
    activity = anonymous_client.app.state.session_store.admin_activity.find_one({"action": "updated_maintenance_mode"})
    assert activity["target_type"] == "runtime_config"
    assert activity["target_id"] == "enabled"

    anonymous_client.post("/auth/logout")
    user_signup = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Runtime Maintenance User",
            "email": user_email,
            "password": "password123",
            "state": "Gujarat",
        },
    )
    assert user_signup.status_code == 200

    blocked_response = anonymous_client.post("/chat", json={"message": "What is Article 21?"})

    assert blocked_response.status_code == 503
    assert blocked_response.json()["detail"] == "The system is temporarily under maintenance. Please try again later."


def test_runtime_maintenance_config_overrides_env_fallback(anonymous_client):
    admin_email = f"runtime-maintenance-override-{uuid.uuid4().hex[:8]}@example.com"
    signup_response = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Runtime Override Admin",
            "email": admin_email,
            "password": "password123",
            "state": "Gujarat",
        },
    )
    assert signup_response.status_code == 200
    assert promote_admin(admin_email)["role"] == "admin"
    anonymous_client.app.state.settings.maintenance_mode = True

    disable_response = anonymous_client.patch("/admin/config/maintenance", json={"enabled": False})
    chat_response = anonymous_client.post("/chat", json={"message": "What is Article 21?"})

    assert disable_response.status_code == 200
    assert disable_response.json()["toggles"]["maintenance_mode"]["enabled"] is False
    assert chat_response.status_code == 200


def test_admin_maintenance_config_patch_blocks_normal_user(anonymous_client):
    email = f"runtime-maintenance-normal-{uuid.uuid4().hex[:8]}@example.com"
    signup_response = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Runtime Maintenance Normal",
            "email": email,
            "password": "password123",
            "state": "Gujarat",
        },
    )
    assert signup_response.status_code == 200

    response = anonymous_client.patch("/admin/config/maintenance", json={"enabled": True})

    assert response.status_code == 403
    assert response.json()["detail"] == "Admin access required"
    assert anonymous_client.app.state.session_store.runtime_config.find_one({"key": "maintenance_mode"}) is None


def test_admin_quality_logs_returns_safe_paginated_filtered_assistant_metadata(anonymous_client):
    admin_email = f"admin-quality-{uuid.uuid4().hex[:8]}@example.com"
    user_email = f"quality-owner-{uuid.uuid4().hex[:8]}@example.com"
    admin_response = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Admin Quality Logs",
            "email": admin_email,
            "password": "password123",
            "state": "Gujarat",
        },
    )
    assert admin_response.status_code == 200
    assert promote_admin(admin_email)["role"] == "admin"
    anonymous_client.post("/auth/logout")

    user_response = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Quality Owner",
            "email": user_email,
            "password": "password123",
            "state": "Maharashtra",
        },
    )
    assert user_response.status_code == 200
    user_id = user_response.json()["user"]["id"]
    store = anonymous_client.app.state.session_store
    session = store.create_session("Quality log chat", user_id=user_id)
    store.add_message(
        session["id"],
        "user",
        "My private complaint text must not appear in quality logs.",
        metadata={"token": "must-not-leak"},
        user_id=user_id,
    )
    store.add_message(
        session["id"],
        "assistant",
        "Fallback answer content must not appear.",
        metadata={
            "route_type": "research",
            "confidence": 0.44,
            "fallback_reason": "retrieval_low_confidence",
            "validation_flags": [],
            "source_sufficiency": {"label": "weak"},
            "auth_session_token": "must-not-leak",
        },
        user_id=user_id,
    )
    store.add_message(
        session["id"],
        "assistant",
        "Unsupported answer content must not appear.",
        metadata={
            "route_classification": {"path": "heavy"},
            "retrieval": {
                "retrieval_confidence": 0.7,
                "fallback_reason_code": "unsupported_output",
                "validation_flags": ["unsupported_output_fallback"],
                "source_sufficiency": {"label": "partial"},
            },
            "google_sub": "must-not-leak",
        },
        user_id=user_id,
    )
    store.add_message(
        session["id"],
        "assistant",
        "Normal answer content must not appear.",
        metadata={
            "route_type": "authority",
            "confidence": 0.92,
            "validation_flags": [],
            "source_sufficiency": {"label": "strong"},
        },
        user_id=user_id,
    )
    anonymous_client.post("/auth/logout")
    login_response = anonymous_client.post(
        "/auth/login",
        json={"email": admin_email, "password": "password123"},
    )
    assert login_response.status_code == 200

    all_response = anonymous_client.get("/admin/quality/logs", params={"skip": 0, "limit": 2})
    assert all_response.status_code == 200
    all_payload = all_response.json()
    assert all_payload["total"] == 3
    assert all_payload["limit"] == 2
    assert len(all_payload["items"]) == 2
    assert {"chat_id", "timestamp", "route_type", "confidence", "fallback_reason", "validation_flags", "source_sufficiency"}.issubset(
        all_payload["items"][0]
    )

    fallback_payload = anonymous_client.get("/admin/quality/logs", params={"fallback_only": "true"}).json()
    assert fallback_payload["total"] == 2
    assert {item["fallback_reason"] for item in fallback_payload["items"]} == {"retrieval_low_confidence", "unsupported_output"}

    low_confidence_payload = anonymous_client.get("/admin/quality/logs", params={"low_confidence_only": "true"}).json()
    assert low_confidence_payload["total"] == 1
    assert low_confidence_payload["items"][0]["confidence"] == 0.44

    unsupported_payload = anonymous_client.get("/admin/quality/logs", params={"unsupported_output_only": "true"}).json()
    assert unsupported_payload["total"] == 1
    assert unsupported_payload["items"][0]["fallback_reason"] == "unsupported_output"
    assert unsupported_payload["items"][0]["validation_flags"] == ["unsupported_output_fallback"]

    serialized = json.dumps([all_payload, fallback_payload, low_confidence_payload, unsupported_payload]).lower()
    forbidden_terms = {
        "private complaint",
        "answer content",
        "quality owner",
        user_email,
        "password_hash",
        "token",
        "auth_session_token",
        "google_sub",
        "auth_provider",
    }
    for forbidden in forbidden_terms:
        assert forbidden not in serialized


def test_admin_direct_answer_content_returns_inventory_without_dataset_contents(anonymous_client):
    email = f"admin-content-{uuid.uuid4().hex[:8]}@example.com"
    signup_response = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Admin Content Viewer",
            "email": email,
            "password": "password123",
            "state": "Gujarat",
        },
    )
    assert signup_response.status_code == 200
    assert promote_admin(email)["role"] == "admin"

    response = anonymous_client.get("/admin/content/direct-answers")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"datasets", "direct_explainer_catalogs"}
    datasets = payload["datasets"]
    assert {"constitution", "ipc", "bns", "bnss"}.issubset(datasets)
    assert datasets["constitution"]["available"] is True
    assert datasets["constitution"]["count"] > 0
    assert datasets["ipc"]["available"] is True
    assert datasets["ipc"]["count"] > 0
    assert datasets["bns"]["available"] is True
    assert datasets["bns"]["count"] > 0
    assert "count" in datasets["bnss"]
    assert "source_names" in datasets["constitution"]
    assert "file_name" in datasets["ipc"]

    catalogs = payload["direct_explainer_catalogs"]
    catalog_names = {catalog["name"] for catalog in catalogs}
    assert {"constitutional_explainers", "general_legal_explainers"}.issubset(catalog_names)
    for catalog in catalogs:
        assert catalog["count"] == len(catalog["items"])
        assert catalog["count"] > 0

    serialized = json.dumps(payload).lower()
    forbidden_terms = [
        "description",
        "text or extracted relevant passage",
        "guarantees equality before the law",
        "cheating and dishonestly inducing",
        "password_hash",
        "token",
        "google_sub",
        "auth_provider",
    ]
    for forbidden in forbidden_terms:
        assert forbidden not in serialized


def test_message_feedback_submission_and_admin_listing_are_safe(anonymous_client):
    admin_email = f"admin-feedback-{uuid.uuid4().hex[:8]}@example.com"
    user_email = f"feedback-user-{uuid.uuid4().hex[:8]}@example.com"
    other_email = f"feedback-other-{uuid.uuid4().hex[:8]}@example.com"
    anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Feedback Admin",
            "email": admin_email,
            "password": "password123",
            "state": "Gujarat",
        },
    )
    assert promote_admin(admin_email)["role"] == "admin"
    anonymous_client.post("/auth/logout")

    user_response = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Feedback User",
            "email": user_email,
            "password": "password123",
            "state": "Delhi",
        },
    )
    user_id = user_response.json()["user"]["id"]
    store = anonymous_client.app.state.session_store
    session = store.create_session("Feedback chat", user_id=user_id)
    user_message = store.add_message(session["id"], "user", "Private user text", user_id=user_id)
    assistant_message = store.add_message(session["id"], "assistant", "Assistant answer text", user_id=user_id)

    not_assistant_response = anonymous_client.post(
        f"/chat/messages/{user_message['id']}/feedback",
        json={"rating": "up", "comment": "ignored"},
    )
    assert not_assistant_response.status_code == 404

    feedback_response = anonymous_client.post(
        f"/chat/messages/{assistant_message['id']}/feedback",
        json={"rating": "down", "comment": "Too generic"},
    )
    assert feedback_response.status_code == 200
    feedback_payload = feedback_response.json()["feedback"]
    assert feedback_payload["chat_id"] == session["id"]
    assert feedback_payload["message_id"] == assistant_message["id"]
    assert feedback_payload["rating"] == "down"
    assert feedback_payload["comment"] == "Too generic"
    assert {"user_id", "email", "password_hash", "token", "auth_provider"}.isdisjoint(feedback_payload)
    second_feedback = anonymous_client.post(
        f"/chat/messages/{assistant_message['id']}/feedback",
        json={"rating": "up"},
    )
    assert second_feedback.status_code == 200

    anonymous_client.post("/auth/logout")
    other_signup = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Other Feedback User",
            "email": other_email,
            "password": "password123",
            "state": "Maharashtra",
        },
    )
    assert other_signup.status_code == 200
    blocked_response = anonymous_client.post(
        f"/chat/messages/{assistant_message['id']}/feedback",
        json={"rating": "up"},
    )
    assert blocked_response.status_code == 404

    anonymous_client.post("/auth/logout")
    login_response = anonymous_client.post(
        "/auth/login",
        json={"email": admin_email, "password": "password123"},
    )
    assert login_response.status_code == 200
    admin_response = anonymous_client.get("/admin/feedback")

    assert admin_response.status_code == 200
    admin_payload = admin_response.json()
    assert admin_payload["total"] == 2
    assert admin_payload["summary"]["total_feedback"] == 2
    assert admin_payload["summary"]["thumbs_up"] == 1
    assert admin_payload["summary"]["thumbs_down"] == 1
    assert admin_payload["summary"]["unchecked"] == 0
    item = admin_payload["items"][0]
    assert item["chat_id"] == session["id"]
    assert item["message_id"] == assistant_message["id"]
    assert item["rating"] in {"up", "down"}
    down_response = anonymous_client.get("/admin/feedback", params={"rating": "down"})
    assert down_response.status_code == 200
    down_payload = down_response.json()
    assert down_payload["total"] == 1
    assert down_payload["rating_filter"] == "down"
    assert down_payload["items"][0]["rating"] == "down"
    assert down_payload["items"][0]["comment"] == "Too generic"
    up_response = anonymous_client.get("/admin/feedback", params={"rating": "up", "recent_limit": 1})
    assert up_response.status_code == 200
    up_payload = up_response.json()
    assert up_payload["recent_limit"] == 1
    assert up_payload["total"] == 1
    assert up_payload["items"][0]["rating"] == "up"
    serialized = json.dumps(admin_payload).lower()
    for forbidden in [
        "feedback user",
        user_email,
        "private user text",
        "assistant answer text",
        "password_hash",
        "token",
        "auth_provider",
    ]:
        assert forbidden not in serialized


def test_admin_activity_logs_safe_read_only_view_events(anonymous_client):
    admin_email = f"admin-activity-{uuid.uuid4().hex[:8]}@example.com"
    user_email = f"activity-user-{uuid.uuid4().hex[:8]}@example.com"
    admin_response = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Activity Admin",
            "email": admin_email,
            "password": "password123",
            "state": "Gujarat",
        },
    )
    assert admin_response.status_code == 200
    admin_id = admin_response.json()["user"]["id"]
    assert promote_admin(admin_email)["role"] == "admin"
    anonymous_client.post("/auth/logout")

    user_response = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Activity User",
            "email": user_email,
            "password": "password123",
            "state": "Delhi",
        },
    )
    assert user_response.status_code == 200
    user_id = user_response.json()["user"]["id"]
    store = anonymous_client.app.state.session_store
    session = store.create_session("Activity visible chat", user_id=user_id)
    store.add_message(session["id"], "assistant", "Safe assistant summary", metadata={"route_type": "authority"}, user_id=user_id)
    anonymous_client.post("/auth/logout")
    login_response = anonymous_client.post(
        "/auth/login",
        json={"email": admin_email, "password": "password123"},
    )
    assert login_response.status_code == 200

    for response in [
        anonymous_client.get("/admin/system/status"),
        anonymous_client.get("/admin/quality/logs"),
        anonymous_client.get("/admin/feedback"),
        anonymous_client.get("/admin/users"),
        anonymous_client.get(f"/admin/users/{user_id}/chats"),
        anonymous_client.get(f"/admin/chats/{session['id']}/messages"),
    ]:
        assert response.status_code == 200

    activity_response = anonymous_client.get("/admin/activity", params={"skip": 0, "limit": 20})

    assert activity_response.status_code == 200
    payload = activity_response.json()
    assert payload["total"] == 6
    assert payload["skip"] == 0
    assert payload["limit"] == 20
    actions = {item["action"] for item in payload["items"]}
    assert {
        "viewed_system_status",
        "viewed_quality_logs",
        "viewed_feedback",
        "viewed_users",
        "viewed_chats",
        "viewed_chat_messages",
    }.issubset(actions)
    for item in payload["items"]:
        assert item["admin_user_id"] == admin_id
        assert item["admin_email"] == admin_email
        assert "created_at" in item
        assert {"password_hash", "token", "auth_provider", "request_body", "content"}.isdisjoint(item)
    chat_event = next(item for item in payload["items"] if item["action"] == "viewed_chat_messages")
    assert chat_event["target_type"] == "chat"
    assert chat_event["target_id"] == str(session["id"])
    serialized = json.dumps(payload).lower()
    for forbidden in ["safe assistant summary", user_email, "password_hash", "auth_provider", "token"]:
        assert forbidden not in serialized


def test_admin_users_supports_search_pagination_and_hides_sensitive_fields(anonymous_client):
    admin_email = f"admin-users-{uuid.uuid4().hex[:8]}@example.com"
    user_email = f"visible-user-{uuid.uuid4().hex[:8]}@example.com"
    anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Visible Admin",
            "email": admin_email,
            "password": "password123",
            "state": "Gujarat",
        },
    )
    assert promote_admin(admin_email)["role"] == "admin"
    anonymous_client.post("/auth/logout")
    anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Visible Search User",
            "email": user_email,
            "password": "password123",
            "state": "Delhi",
        },
    )
    anonymous_client.post("/auth/logout")
    login_response = anonymous_client.post(
        "/auth/login",
        json={"email": admin_email, "password": "password123"},
    )
    assert login_response.status_code == 200

    response = anonymous_client.get("/admin/users", params={"search": "Visible Search", "skip": 0, "limit": 5})

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["skip"] == 0
    assert payload["limit"] == 5
    assert payload["items"][0]["email"] == user_email
    assert payload["items"][0]["role"] == "user"
    assert payload["items"][0]["status"] == "active"
    forbidden_fields = {"password_hash", "token", "token_hash", "google_sub", "auth_provider"}
    assert forbidden_fields.isdisjoint(payload["items"][0])


def test_admin_can_block_and_unblock_normal_user_and_activity_is_logged(anonymous_client):
    admin_email = f"status-admin-{uuid.uuid4().hex[:8]}@example.com"
    user_email = f"status-user-{uuid.uuid4().hex[:8]}@example.com"
    admin_signup = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Status Admin",
            "email": admin_email,
            "password": "password123",
            "state": "Gujarat",
        },
    )
    assert admin_signup.status_code == 200
    assert promote_admin(admin_email)["role"] == "admin"
    anonymous_client.post("/auth/logout")
    user_signup = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Status User",
            "email": user_email,
            "password": "password123",
            "state": "Delhi",
        },
    )
    assert user_signup.status_code == 200
    user_id = user_signup.json()["user"]["id"]
    anonymous_client.post("/auth/logout")
    assert anonymous_client.post("/auth/login", json={"email": admin_email, "password": "password123"}).status_code == 200

    block_response = anonymous_client.patch(f"/admin/users/{user_id}/status", json={"status": "blocked"})

    assert block_response.status_code == 200
    blocked_user = block_response.json()["user"]
    assert blocked_user["id"] == user_id
    assert blocked_user["status"] == "blocked"
    assert {"password_hash", "token", "token_hash", "google_sub", "auth_provider"}.isdisjoint(blocked_user)
    stored_document = anonymous_client.app.state.session_store.users.find_one({"id": user_id})
    assert stored_document["status"] == "blocked"
    activity = anonymous_client.app.state.session_store.admin_activity.find_one({"action": "updated_user_status_blocked"})
    assert activity["target_type"] == "user"
    assert activity["target_id"] == str(user_id)

    anonymous_client.post("/auth/logout")
    assert anonymous_client.post("/auth/login", json={"email": user_email, "password": "password123"}).status_code == 200
    history_response = anonymous_client.get("/chat/history")
    chat_response = anonymous_client.post("/chat", json={"message": "What is Article 21?"})

    assert history_response.status_code == 403
    assert history_response.json()["detail"] == "Your account is blocked. Please contact an administrator."
    assert chat_response.status_code == 403
    assert chat_response.json()["detail"] == "Your account is blocked. Please contact an administrator."

    anonymous_client.post("/auth/logout")
    assert anonymous_client.post("/auth/login", json={"email": admin_email, "password": "password123"}).status_code == 200
    unblock_response = anonymous_client.patch(f"/admin/users/{user_id}/status", json={"status": "active"})

    assert unblock_response.status_code == 200
    assert unblock_response.json()["user"]["status"] == "active"
    assert anonymous_client.app.state.session_store.admin_activity.find_one({"action": "updated_user_status_active"}) is not None

    anonymous_client.post("/auth/logout")
    assert anonymous_client.post("/auth/login", json={"email": user_email, "password": "password123"}).status_code == 200
    assert anonymous_client.get("/chat/history").status_code == 200


def test_admin_user_status_update_blocks_self_admin_targets_invalid_status_and_normal_users(anonymous_client):
    admin_email = f"status-guard-admin-{uuid.uuid4().hex[:8]}@example.com"
    second_admin_email = f"status-guard-second-{uuid.uuid4().hex[:8]}@example.com"
    normal_email = f"status-guard-normal-{uuid.uuid4().hex[:8]}@example.com"
    admin_signup = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Status Guard Admin",
            "email": admin_email,
            "password": "password123",
            "state": "Gujarat",
        },
    )
    assert admin_signup.status_code == 200
    admin_id = admin_signup.json()["user"]["id"]
    assert promote_admin(admin_email)["role"] == "admin"
    anonymous_client.post("/auth/logout")
    second_admin_signup = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Second Status Admin",
            "email": second_admin_email,
            "password": "password123",
            "state": "Gujarat",
        },
    )
    assert second_admin_signup.status_code == 200
    second_admin_id = second_admin_signup.json()["user"]["id"]
    assert promote_admin(second_admin_email)["role"] == "admin"
    anonymous_client.post("/auth/logout")
    normal_signup = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Status Guard Normal",
            "email": normal_email,
            "password": "password123",
            "state": "Delhi",
        },
    )
    assert normal_signup.status_code == 200
    normal_id = normal_signup.json()["user"]["id"]

    normal_block_attempt = anonymous_client.patch(f"/admin/users/{admin_id}/status", json={"status": "blocked"})
    assert normal_block_attempt.status_code == 403
    assert normal_block_attempt.json()["detail"] == "Admin access required"

    anonymous_client.post("/auth/logout")
    assert anonymous_client.post("/auth/login", json={"email": admin_email, "password": "password123"}).status_code == 200

    self_response = anonymous_client.patch(f"/admin/users/{admin_id}/status", json={"status": "blocked"})
    admin_target_response = anonymous_client.patch(f"/admin/users/{second_admin_id}/status", json={"status": "blocked"})
    invalid_status_response = anonymous_client.patch(f"/admin/users/{normal_id}/status", json={"status": "disabled"})

    assert self_response.status_code == 400
    assert self_response.json()["detail"] == "Admins cannot block themselves"
    assert admin_target_response.status_code == 400
    assert admin_target_response.json()["detail"] == "Admin users cannot be blocked"
    assert invalid_status_response.status_code == 400
    assert invalid_status_response.json()["detail"] == "User status must be active or blocked"
    assert anonymous_client.app.state.session_store.users.find_one({"id": second_admin_id})["status"] == "active"


def test_admin_user_chats_blocks_authenticated_normal_user(client):
    response = client.get("/admin/users/1/chats")

    assert response.status_code == 403
    assert response.json()["detail"] == "Admin access required"


def test_admin_chat_inspection_allows_admin_and_sanitizes_metadata(anonymous_client):
    admin_email = f"admin-chat-{uuid.uuid4().hex[:8]}@example.com"
    user_email = f"chat-owner-{uuid.uuid4().hex[:8]}@example.com"
    admin_response = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Admin Chat Inspector",
            "email": admin_email,
            "password": "password123",
            "state": "Gujarat",
        },
    )
    assert admin_response.status_code == 200
    assert promote_admin(admin_email)["role"] == "admin"
    anonymous_client.post("/auth/logout")

    user_response = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Chat Owner",
            "email": user_email,
            "password": "password123",
            "state": "Maharashtra",
        },
    )
    assert user_response.status_code == 200
    user_id = user_response.json()["user"]["id"]
    store = anonymous_client.app.state.session_store
    session = store.create_session("Admin visible chat", user_id=user_id)
    store.add_message(
        session["id"],
        "user",
        "What is section 420 IPC?",
        metadata={
            "route_type": "authority",
            "confidence": 0.92,
            "token": "must-not-leak",
            "password_hash": "must-not-leak",
        },
        user_id=user_id,
    )
    store.add_message(
        session["id"],
        "assistant",
        "Section 420 IPC concerns cheating.",
        metadata={
            "route_type": "authority",
            "confidence": 0.88,
            "fallback_reason": "local_dataset",
            "validation_flags": ["unsupported_output_fallback"],
            "disclaimer_mode": "high_risk",
            "source_sufficiency": {"label": "weak", "reasons": ["confidence_low"]},
            "auth_session_token": "must-not-leak",
            "google_sub": "must-not-leak",
        },
        user_id=user_id,
    )
    anonymous_client.post("/auth/logout")
    login_response = anonymous_client.post(
        "/auth/login",
        json={"email": admin_email, "password": "password123"},
    )
    assert login_response.status_code == 200

    chats_response = anonymous_client.get(f"/admin/users/{user_id}/chats")
    assert chats_response.status_code == 200
    chats_payload = chats_response.json()
    assert chats_payload["total"] == 1
    chat_item = chats_payload["items"][0]
    assert chat_item["id"] == session["id"]
    assert chat_item["user_id"] == user_id
    assert chat_item["message_count"] == 2
    assert chat_item["metadata"]["route_type"] == "authority"
    assert chat_item["metadata"]["confidence"] == 0.88
    assert chat_item["metadata"]["fallback_reason"] == "local_dataset"

    messages_response = anonymous_client.get(f"/admin/chats/{session['id']}/messages")
    assert messages_response.status_code == 200
    messages_payload = messages_response.json()
    assert messages_payload["chat"]["id"] == session["id"]
    assert messages_payload["chat"]["user_id"] == user_id
    assert messages_payload["total"] == 2
    assert messages_payload["items"][0]["metadata"]["route_type"] == "authority"
    assert messages_payload["items"][0]["metadata"]["confidence"] == 0.92
    assert messages_payload["items"][1]["metadata"]["fallback_reason"] == "local_dataset"
    assert messages_payload["items"][1]["metadata"]["validation_flags"] == ["unsupported_output_fallback"]
    assert messages_payload["items"][1]["metadata"]["disclaimer_mode"] == "high_risk"
    assert messages_payload["items"][1]["metadata"]["source_sufficiency"]["label"] == "weak"

    forbidden_fields = {"password_hash", "token", "token_hash", "auth_session_token", "google_sub", "auth_provider"}
    assert forbidden_fields.isdisjoint(chat_item["metadata"])
    for message in messages_payload["items"]:
        assert forbidden_fields.isdisjoint(message)
        assert forbidden_fields.isdisjoint(message["metadata"])


def test_admin_chat_inspection_returns_404_for_missing_user_and_chat(anonymous_client):
    email = f"admin-missing-chat-{uuid.uuid4().hex[:8]}@example.com"
    signup_response = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Admin Missing Chat",
            "email": email,
            "password": "password123",
            "state": "Gujarat",
        },
    )
    assert signup_response.status_code == 200
    assert promote_admin(email)["role"] == "admin"

    missing_user_response = anonymous_client.get("/admin/users/999999/chats")
    assert missing_user_response.status_code == 404
    assert missing_user_response.json()["detail"] == "User not found"

    missing_chat_response = anonymous_client.get("/admin/chats/999999/messages")
    assert missing_chat_response.status_code == 404
    assert missing_chat_response.json()["detail"] == "Chat not found"


def test_google_callback_sets_session_cookie(monkeypatch, anonymous_client):
    anonymous_client.app.state.settings.google_client_id = "google-client"
    anonymous_client.app.state.settings.google_client_secret = "google-secret"
    anonymous_client.app.state.settings.app_base_url = "https://testserver"
    anonymous_client.app.state.settings.google_redirect_uri = "https://testserver/auth/google/callback"

    def fake_exchange(request, code):
        assert code == "auth-code"
        assert auth_routes.google_redirect_uri(request) == "https://testserver/auth/google/callback"
        return {"access_token": "google-access-token"}

    def fake_userinfo(access_token):
        assert access_token == "google-access-token"
        return {
            "sub": "google-sub-123",
            "email": "google@example.com",
            "email_verified": True,
            "name": "Google User",
        }

    monkeypatch.setattr(auth_routes, "exchange_google_code_for_tokens", fake_exchange)
    monkeypatch.setattr(auth_routes, "fetch_google_userinfo", fake_userinfo)

    login_redirect = anonymous_client.get("/auth/google/login", follow_redirects=False)
    assert login_redirect.status_code == 307

    redirect_url = login_redirect.headers["location"]
    redirect_query = parse_qs(urlparse(redirect_url).query)
    assert redirect_query["redirect_uri"][0] == "https://testserver/auth/google/callback"
    state = redirect_query["state"][0]

    callback_response = anonymous_client.get(
        f"https://testserver/auth/google/callback?code=auth-code&state={state}",
        follow_redirects=False,
    )
    assert callback_response.status_code == 307
    assert callback_response.headers["location"] == "https://testserver/frontend/index.html"
    set_cookie = callback_response.headers.get("set-cookie", "")
    assert "session_token=" in set_cookie
    assert "HttpOnly" in set_cookie
    assert "Secure" in set_cookie
    assert "SameSite=none" in set_cookie

    me_response = anonymous_client.get("https://testserver/auth/me", headers={"Authorization": "Bearer stale-localstorage-token"})
    assert me_response.status_code == 200
    assert me_response.json()["email"] == "google@example.com"


def test_google_callback_failure_redirects_to_auth_with_error(anonymous_client):
    anonymous_client.app.state.settings.app_base_url = "https://testserver"

    response = anonymous_client.get(
        "https://testserver/auth/google/callback?error=access_denied",
        follow_redirects=False,
    )

    assert response.status_code == 307
    assert response.headers["location"] == "https://testserver/frontend/auth.html?error=google_auth_failed"


def test_password_reset_requires_smtp_configuration(anonymous_client):
    anonymous_client.app.state.settings.smtp_host = ""
    anonymous_client.app.state.settings.smtp_username = ""
    anonymous_client.app.state.settings.smtp_password = ""
    anonymous_client.app.state.settings.smtp_from_email = ""

    response = anonymous_client.post("/auth/password-reset", json={"email": "missing@example.com"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Password reset email is not configured yet."


def test_password_reset_email_has_html_template_and_plain_text_fallback():
    settings = Settings(
        SMTP_HOST="smtp.example.com",
        SMTP_USERNAME="sender@example.com",
        SMTP_PASSWORD="app-password",
        SMTP_FROM_EMAIL="sender@example.com",
        PASSWORD_RESET_TOKEN_TTL_MINUTES=45,
    )
    mailer = SmtpMailer(settings)
    reset_url = "https://example.com/frontend/auth.html?reset_token=abc123"

    text_body = mailer._build_password_reset_text(reset_url)
    html_body = mailer._build_password_reset_html(reset_url)

    assert reset_url in text_body
    assert "This link expires in 45 minutes." in text_body
    assert "{{RESET_LINK}}" not in text_body
    assert "<h1" in html_body
    assert "Reset password" in html_body
    assert f'href="{reset_url}"' in html_body
    assert "If the button does not work" in html_body
    assert "Your current password will remain unchanged" in html_body
    assert "{{RESET_LINK}}" not in html_body


def test_password_reset_request_and_confirm_flow(monkeypatch, anonymous_client):
    sent_messages: list[tuple[str, str]] = []

    anonymous_client.app.state.settings.smtp_host = "smtp.gmail.com"
    anonymous_client.app.state.settings.smtp_username = "sender@example.com"
    anonymous_client.app.state.settings.smtp_password = "app-password"
    anonymous_client.app.state.settings.smtp_from_email = "sender@example.com"
    anonymous_client.app.state.settings.app_base_url = "http://testserver"

    anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Reset User",
            "email": "reset@example.com",
            "password": "oldpassword123",
            "state": "Gujarat",
        },
    )
    anonymous_client.post("/auth/logout")

    def fake_send_password_reset_email(self, recipient_email: str, reset_url: str) -> None:
        sent_messages.append((recipient_email, reset_url))

    monkeypatch.setattr(SmtpMailer, "send_password_reset_email", fake_send_password_reset_email)

    request_response = anonymous_client.post("/auth/password-reset", json={"email": "reset@example.com"})
    assert request_response.status_code == 200
    assert sent_messages

    recipient_email, reset_url = sent_messages[0]
    assert recipient_email == "reset@example.com"
    token = parse_qs(urlparse(reset_url).query)["reset_token"][0]

    confirm_response = anonymous_client.post(
        "/auth/password-reset/confirm",
        json={"token": token, "password": "newpassword123"},
    )
    assert confirm_response.status_code == 200
    assert confirm_response.json()["message"] == "Your password has been reset. Please log in with your new password."

    reused_response = anonymous_client.post(
        "/auth/password-reset/confirm",
        json={"token": token, "password": "anotherpassword123"},
    )
    assert reused_response.status_code == 400
    assert reused_response.json()["detail"] == "Reset link is invalid or has expired."

    old_login = anonymous_client.post(
        "/auth/login",
        json={"email": "reset@example.com", "password": "oldpassword123"},
    )
    assert old_login.status_code == 401

    new_login = anonymous_client.post(
        "/auth/login",
        json={"email": "reset@example.com", "password": "newpassword123"},
    )
    assert new_login.status_code == 200


def test_signup_cookie_not_secure_in_debug_local_mode(anonymous_client):
    email = f"cookie-debug-{uuid.uuid4().hex[:8]}@example.com"
    response = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Cookie Debug User",
            "email": email,
            "password": "password123",
            "state": "Gujarat",
        },
    )
    assert response.status_code == 200
    set_cookie = response.headers.get("set-cookie", "")
    assert "Secure" not in set_cookie


def test_signup_cookie_secure_for_https_forwarded_requests(monkeypatch, anonymous_client):
    anonymous_client.app.state.settings.debug = False
    email = f"cookie-secure-{uuid.uuid4().hex[:8]}@example.com"
    response = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Cookie Secure User",
            "email": email,
            "password": "password123",
            "state": "Gujarat",
        },
        headers={"x-forwarded-proto": "https"},
    )
    assert response.status_code == 200
    set_cookie = response.headers.get("set-cookie", "")
    assert "Secure" in set_cookie
