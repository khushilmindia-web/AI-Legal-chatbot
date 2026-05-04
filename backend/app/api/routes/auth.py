from __future__ import annotations

import logging
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
    PasswordResetConfirmRequest,
    PasswordResetRequest,
)
from backend.app.services.mailer import MailDeliveryError, SmtpMailer


router = APIRouter(tags=["auth"])
logger = logging.getLogger(__name__)

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
GOOGLE_SCOPES = "openid email profile"
GOOGLE_STATE_TTL_SECONDS = 600
GOOGLE_OAUTH_STATE_CACHE: dict[str, float] = {}


def google_oauth_http_request(method: str, url: str, **kwargs):
    with requests.Session() as session:
        session.trust_env = False
        return session.request(method=method, url=url, **kwargs)


def normalize_email(email: str) -> str:
    return email.strip().lower()


def external_base_url(request: Request) -> str:
    configured = request.app.state.settings.app_base_url.strip()
    if configured:
        return configured.rstrip("/")
    return str(request.base_url).rstrip("/")


def build_frontend_url(request: Request, path: str, **query: str) -> str:
    base_url = external_base_url(request)
    cleaned_path = path if path.startswith("/") else f"/{path}"
    if not query:
        return f"{base_url}{cleaned_path}"
    return f"{base_url}{cleaned_path}?{urlencode(query)}"


def frontend_auth_url(request: Request, **query: str) -> str:
    return build_frontend_url(request, request.app.state.settings.frontend_auth_path, **query)


def frontend_app_url(request: Request) -> str:
    return build_frontend_url(request, request.app.state.settings.frontend_app_path)


def google_redirect_uri(request: Request) -> str:
    settings = request.app.state.settings
    redirect_uri = settings.google_oauth_redirect_uri.strip()
    configured = settings.google_redirect_uri.strip()
    if configured and configured.rstrip("/") != redirect_uri.rstrip("/"):
        logger.warning(
            "GOOGLE_REDIRECT_URI ignored because APP_BASE_URL is authoritative configured=%s effective=%s",
            configured,
            redirect_uri,
        )
    if redirect_uri.startswith("http://") and "localhost" not in redirect_uri and "127.0.0.1" not in redirect_uri:
        logger.warning("Google OAuth redirect_uri is not HTTPS; ngrok Google OAuth requires exact HTTPS URL redirect_uri=%s", redirect_uri)
    return redirect_uri or build_frontend_url(request, "/auth/google/callback")


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
    redirect_uri = google_redirect_uri(request)
    logger.info("Google OAuth token exchange redirect_uri=%s", redirect_uri)
    response = google_oauth_http_request(
        "POST",
        GOOGLE_TOKEN_URL,
        data={
            "code": code,
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
        timeout=20,
    )
    if not response.ok:
        logger.warning(
            "Google token exchange failed status=%s redirect_uri=%s body=%s",
            response.status_code,
            redirect_uri,
            response.text[:500],
        )
    response.raise_for_status()
    return response.json()


def fetch_google_userinfo(access_token: str) -> dict:
    response = google_oauth_http_request(
        "GET",
        GOOGLE_USERINFO_URL,
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=20,
    )
    if not response.ok:
        logger.warning(
            "Google userinfo fetch failed status=%s body=%s",
            response.status_code,
            response.text[:500],
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
        logger.info(
            "Google OAuth Mongo user lookup result=existing_google user_id=%s email=%s",
            existing_google_user.get("id"),
            existing_google_user.get("email"),
        )
        return existing_google_user

    existing_email_user = store.get_user_by_email(email)
    if existing_email_user:
        linked_user = store.link_google_account(existing_email_user["id"], google_sub)
        if linked_user:
            logger.info(
                "Google OAuth Mongo user lookup result=linked_existing_email user_id=%s email=%s",
                linked_user.get("id"),
                linked_user.get("email"),
            )
            return linked_user
        raise ValueError("Failed to link Google account")

    user = store.create_google_user(full_name=full_name, email=email, google_sub=google_sub, state=None)
    logger.info(
        "Google OAuth Mongo user lookup result=created_google user_id=%s email=%s",
        user.get("id"),
        user.get("email"),
    )
    return user


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
    logger.info("/auth/me authenticated user_id=%s email=%s", user.get("id"), user.get("email"))
    return AuthUser(**user)


@router.post("/auth/logout", response_model=MessageResponse)
def logout(request: Request, response: Response) -> MessageResponse:
    token = get_request_token(request)
    if token:
        request.app.state.session_store.delete_auth_session(token)
    clear_auth_cookie(response, request)
    return MessageResponse(message="Logged out successfully.")


@router.post("/auth/password-reset", response_model=MessageResponse)
def password_reset(payload: PasswordResetRequest, request: Request) -> MessageResponse:
    settings = request.app.state.settings
    if not settings.smtp_configured:
        raise HTTPException(status_code=503, detail="Password reset email is not configured yet.")

    email = normalize_email(payload.email)
    user = request.app.state.session_store.get_user_by_email(email)
    if user:
        token = request.app.state.session_store.create_password_reset_token(
            user_id=user["id"],
            ttl_minutes=settings.password_reset_token_ttl_minutes,
        )
        reset_url = SmtpMailer(settings).build_password_reset_url(token)
        try:
            SmtpMailer(settings).send_password_reset_email(recipient_email=user["email"], reset_url=reset_url)
        except MailDeliveryError as exc:
            raise HTTPException(status_code=502, detail="Password reset email could not be sent.") from exc

    return MessageResponse(message="If an account with this email exists, a password reset link has been sent.")


@router.post("/auth/password-reset/confirm", response_model=MessageResponse)
def password_reset_confirm(payload: PasswordResetConfirmRequest, request: Request) -> MessageResponse:
    token = payload.token.strip()
    password = payload.password.strip()
    if not token:
        raise HTTPException(status_code=400, detail="Reset token is required.")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    user = request.app.state.session_store.reset_password_with_token(token=token, new_password=password)
    if not user:
        raise HTTPException(status_code=400, detail="Reset link is invalid or has expired.")
    return MessageResponse(message="Your password has been reset. Please log in with your new password.")


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
    redirect_uri = google_redirect_uri(request)
    logger.info(
        "Google OAuth login APP_BASE_URL=%s redirect_uri=%s",
        request.app.state.settings.app_base_url,
        redirect_uri,
    )
    params = {
        "client_id": request.app.state.settings.google_client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": GOOGLE_SCOPES,
        "state": state,
        "access_type": "offline",
        "prompt": "select_account",
    }
    return RedirectResponse(url=f"{GOOGLE_AUTH_URL}?{urlencode(params)}")


@router.get("/auth/google/callback")
def google_callback(request: Request, code: str | None = None, state: str | None = None, error: str | None = None):
    logger.info(
        "Google OAuth callback reached has_code=%s has_state=%s error=%s cookies=%s",
        bool(code),
        bool(state),
        error or "",
        sorted(request.cookies.keys()),
    )
    if error:
        logger.warning("Google OAuth callback received provider error=%s", error)
        return RedirectResponse(url=frontend_auth_url(request, error="google_auth_failed"))
    if not code or not state:
        logger.warning("Google OAuth callback incomplete has_code=%s has_state=%s", bool(code), bool(state))
        return RedirectResponse(url=frontend_auth_url(request, error="google_auth_incomplete"))

    cleanup_google_states()
    if state not in GOOGLE_OAUTH_STATE_CACHE:
        logger.warning("Google OAuth callback invalid_state state_prefix=%s", state[:8] if state else "")
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
        redirect_target = frontend_app_url(request)
        redirect = RedirectResponse(url=redirect_target)
        set_auth_cookie(redirect, request, app_token, secure=True, samesite="none")
        logger.info(
            "Google OAuth callback auth_session created user_id=%s token_prefix=%s redirect=%s cookie_name=%s",
            user["id"],
            app_token[:8],
            redirect_target,
            request.app.state.settings.auth_cookie_name,
        )
        return redirect
    except requests.Timeout:
        logger.warning("Google OAuth callback timed out redirect_uri=%s", google_redirect_uri(request))
        return RedirectResponse(url=frontend_auth_url(request, error="google_auth_timeout"))
    except requests.HTTPError as exc:
        logger.warning("Google OAuth callback HTTP error: %s", exc)
        return RedirectResponse(url=frontend_auth_url(request, error="google_auth_config_invalid"))
    except requests.RequestException:
        logger.warning("Google OAuth callback request exception", exc_info=True)
        return RedirectResponse(url=frontend_auth_url(request, error="google_auth_request_failed"))
    except Exception:
        logger.exception("Google OAuth callback failed unexpectedly")
        return RedirectResponse(url=frontend_auth_url(request, error="google_auth_failed"))
