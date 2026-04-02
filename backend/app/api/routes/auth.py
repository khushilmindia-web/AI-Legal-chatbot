from __future__ import annotations

import secrets
import time
from urllib.parse import urlencode

import requests
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import RedirectResponse

from backend.app.api.auth_utils import (
    clear_auth_cookie,
    get_request_token,
    require_current_user,
    set_auth_cookie,
)
from backend.app.models.schemas import (
    AuthLoginRequest,
    AuthResponse,
    AuthSignupRequest,
    AuthUser,
    GoogleAuthStatusResponse,
    MessageResponse,
    PasswordResetRequest,
)


router = APIRouter(tags=["auth"])

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
GOOGLE_SCOPES = "openid email profile"
GOOGLE_STATE_TTL_SECONDS = 600
GOOGLE_OAUTH_STATE_CACHE: dict[str, float] = {}


def normalize_email(email: str) -> str:
    return email.strip().lower()


def build_frontend_url(request: Request, path: str, **query: str) -> str:
    base_url = str(request.base_url).rstrip("/")
    cleaned_path = path if path.startswith("/") else f"/{path}"
    if not query:
        return f"{base_url}{cleaned_path}"
    return f"{base_url}{cleaned_path}?{urlencode(query)}"


def frontend_auth_url(request: Request, **query: str) -> str:
    return build_frontend_url(request, request.app.state.settings.frontend_auth_path, **query)


def frontend_app_url(request: Request) -> str:
    return build_frontend_url(request, request.app.state.settings.frontend_app_path)


def google_redirect_uri(request: Request) -> str:
    configured = request.app.state.settings.google_redirect_uri.strip()
    if configured:
        return configured
    return build_frontend_url(request, "/auth/google/callback")


def google_auth_configured(request: Request) -> bool:
    settings = request.app.state.settings
    return bool(settings.google_client_id.strip() and settings.google_client_secret.strip())


def cleanup_google_states() -> None:
    cutoff = time.time() - GOOGLE_STATE_TTL_SECONDS
    stale_keys = [key for key, created_at in GOOGLE_OAUTH_STATE_CACHE.items() if created_at < cutoff]
    for key in stale_keys:
        GOOGLE_OAUTH_STATE_CACHE.pop(key, None)


def exchange_google_code_for_tokens(request: Request, code: str) -> dict:
    settings = request.app.state.settings
    response = requests.post(
        GOOGLE_TOKEN_URL,
        data={
            "code": code,
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "redirect_uri": google_redirect_uri(request),
            "grant_type": "authorization_code",
        },
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def fetch_google_userinfo(access_token: str) -> dict:
    response = requests.get(
        GOOGLE_USERINFO_URL,
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=20,
    )
    response.raise_for_status()
    profile = response.json()
    if not profile.get("sub") or not profile.get("email"):
        raise ValueError("Google profile missing required fields")
    if profile.get("email_verified") is not True:
        raise ValueError("Google email is not verified")
    return profile


def get_or_create_google_auth_user(request: Request, profile: dict):
    store = request.app.state.session_store
    google_sub = profile["sub"]
    email = normalize_email(profile["email"])
    full_name = (profile.get("name") or email.split("@")[0]).strip()

    existing_google_user = store.get_user_by_google_sub(google_sub)
    if existing_google_user:
        return existing_google_user

    existing_email_user = store.get_user_by_email(email)
    if existing_email_user:
        linked_user = store.link_google_account(existing_email_user["id"], google_sub)
        if linked_user:
            return linked_user
        raise ValueError("Failed to link Google account")

    return store.create_google_user(full_name=full_name, email=email, google_sub=google_sub, state=None)


@router.post("/auth/signup", response_model=AuthResponse)
def signup(payload: AuthSignupRequest, request: Request, response: Response) -> AuthResponse:
    store = request.app.state.session_store
    full_name = payload.full_name.strip()
    email = normalize_email(payload.email)
    password = payload.password.strip()
    state = payload.state.strip() if payload.state and payload.state.strip() else None

    if len(full_name) < 2:
        raise HTTPException(status_code=400, detail="Full name must be at least 2 characters")
    if "@" not in email or "." not in email:
        raise HTTPException(status_code=400, detail="Enter a valid email address")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    if store.get_user_by_email(email):
        raise HTTPException(status_code=409, detail="An account with this email already exists")

    user = store.create_user(full_name=full_name, email=email, password=password, state=state)
    token = store.create_auth_session(user["id"], request.app.state.settings.auth_session_duration_days)
    set_auth_cookie(response, request, token)
    return AuthResponse(token=token, user=AuthUser(**user))


@router.post("/auth/login", response_model=AuthResponse)
def login(payload: AuthLoginRequest, request: Request, response: Response) -> AuthResponse:
    store = request.app.state.session_store
    user = store.authenticate_user(payload.email, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = store.create_auth_session(user["id"], request.app.state.settings.auth_session_duration_days)
    set_auth_cookie(response, request, token)
    return AuthResponse(token=token, user=AuthUser(**user))


@router.get("/auth/me", response_model=AuthUser)
def me(request: Request) -> AuthUser:
    user = require_current_user(request)
    return AuthUser(**user)


@router.post("/auth/logout", response_model=MessageResponse)
def logout(request: Request, response: Response) -> MessageResponse:
    token = get_request_token(request)
    if token:
        request.app.state.session_store.delete_auth_session(token)
    clear_auth_cookie(response, request)
    return MessageResponse(message="Logged out successfully.")


@router.post("/auth/password-reset", response_model=MessageResponse)
def password_reset(payload: PasswordResetRequest) -> MessageResponse:
    _ = normalize_email(payload.email)
    return MessageResponse(message="If an account with this email exists, a password reset link has been sent.")


@router.get("/auth/google/status", response_model=GoogleAuthStatusResponse)
def google_auth_status(request: Request) -> GoogleAuthStatusResponse:
    settings = request.app.state.settings
    return GoogleAuthStatusResponse(
        configured=google_auth_configured(request),
        client_id_configured=bool(settings.google_client_id.strip()),
        gis_script_required=True,
    )


@router.get("/auth/google/login")
def google_login(request: Request):
    if not google_auth_configured(request):
        return RedirectResponse(url=frontend_auth_url(request, error="google_auth_not_configured"))

    cleanup_google_states()
    state = secrets.token_urlsafe(24)
    GOOGLE_OAUTH_STATE_CACHE[state] = time.time()
    params = {
        "client_id": request.app.state.settings.google_client_id,
        "redirect_uri": google_redirect_uri(request),
        "response_type": "code",
        "scope": GOOGLE_SCOPES,
        "state": state,
        "access_type": "offline",
        "prompt": "select_account",
    }
    return RedirectResponse(url=f"{GOOGLE_AUTH_URL}?{urlencode(params)}")


@router.get("/auth/google/callback")
def google_callback(request: Request, code: str | None = None, state: str | None = None, error: str | None = None):
    if error:
        return RedirectResponse(url=frontend_auth_url(request, error="google_auth_failed"))
    if not code or not state:
        return RedirectResponse(url=frontend_auth_url(request, error="google_auth_incomplete"))

    cleanup_google_states()
    if state not in GOOGLE_OAUTH_STATE_CACHE:
        return RedirectResponse(url=frontend_auth_url(request, error="google_auth_state_invalid"))
    GOOGLE_OAUTH_STATE_CACHE.pop(state, None)

    try:
        token_payload = exchange_google_code_for_tokens(request, code)
        access_token = token_payload.get("access_token", "")
        if not access_token:
            raise ValueError("Missing Google access token")
        google_profile = fetch_google_userinfo(access_token)
        user = get_or_create_google_auth_user(request, google_profile)
        app_token = request.app.state.session_store.create_auth_session(
            user["id"],
            request.app.state.settings.auth_session_duration_days,
        )
        redirect = RedirectResponse(url=frontend_app_url(request))
        set_auth_cookie(redirect, request, app_token)
        return redirect
    except requests.Timeout:
        return RedirectResponse(url=frontend_auth_url(request, error="google_auth_timeout"))
    except requests.RequestException:
        return RedirectResponse(url=frontend_auth_url(request, error="google_auth_request_failed"))
    except Exception:
        return RedirectResponse(url=frontend_auth_url(request, error="google_auth_failed"))
