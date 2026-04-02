from __future__ import annotations

from urllib.parse import parse_qs, urlparse

from backend.app.api.routes import auth as auth_routes


def test_chat_routes_require_auth(anonymous_client):
    response = anonymous_client.get("/chat/history")

    assert response.status_code == 401
    assert response.json()["detail"] == "Authentication required"


def test_signup_me_logout_and_protected_redirects(anonymous_client):
    signup_response = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Integration User",
            "email": "integration@example.com",
            "password": "password123",
            "state": "Gujarat",
        },
    )

    assert signup_response.status_code == 200
    assert signup_response.json()["user"]["email"] == "integration@example.com"

    me_response = anonymous_client.get("/auth/me")
    assert me_response.status_code == 200
    assert me_response.json()["full_name"] == "Integration User"

    root_response = anonymous_client.get("/", follow_redirects=False)
    assert root_response.status_code == 307
    assert root_response.headers["location"] == "/frontend/index.html"

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
    anonymous_client.app.state.settings.google_redirect_uri = "http://testserver/auth/google/callback"

    def fake_exchange(request, code):
        assert code == "auth-code"
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
    state = parse_qs(urlparse(redirect_url).query)["state"][0]

    callback_response = anonymous_client.get(
        f"/auth/google/callback?code=auth-code&state={state}",
        follow_redirects=False,
    )
    assert callback_response.status_code == 307
    assert callback_response.headers["location"] == "http://testserver/frontend/index.html"

    me_response = anonymous_client.get("/auth/me")
    assert me_response.status_code == 200
    assert me_response.json()["email"] == "google@example.com"
