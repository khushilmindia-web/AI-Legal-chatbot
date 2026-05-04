from __future__ import annotations

import uuid
from urllib.parse import parse_qs, urlparse

from backend.app.core.config import Settings
from backend.app.api.routes import auth as auth_routes
from backend.app.services.mailer import SmtpMailer


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
