from __future__ import annotations

import logging

from fastapi import HTTPException, Request, Response


logger = logging.getLogger(__name__)


def get_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return authorization.strip()


def get_request_token(request: Request) -> str | None:
    cookie_name = request.app.state.settings.auth_cookie_name
    cookie_token = request.cookies.get(cookie_name)
    if cookie_token:
        logger.debug(
            "auth token source=cookie cookie_name=%s token_prefix=%s",
            cookie_name,
            cookie_token[:8],
        )
        return cookie_token

    authorization = request.headers.get("Authorization")
    bearer_token = get_bearer_token(authorization)
    if bearer_token:
        logger.debug("auth token source=authorization token_prefix=%s", bearer_token[:8])
        return bearer_token

    logger.debug(
        "auth token missing cookie_name=%s available_cookies=%s has_authorization=%s",
        cookie_name,
        sorted(request.cookies.keys()),
        bool(authorization),
    )
    return None


def get_optional_current_user(request: Request):
    token = get_request_token(request)
    if not token:
        logger.debug("auth/me session lookup no token cookie_name=%s", request.app.state.settings.auth_cookie_name)
        return None
    user = request.app.state.session_store.get_user_by_token(token)
    logger.debug(
        "auth/me session lookup result=%s cookie_name=%s token_prefix=%s",
        "hit" if user else "miss",
        request.app.state.settings.auth_cookie_name,
        token[:8],
    )
    return user


def require_current_user(request: Request):
    user = get_optional_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


def set_auth_cookie(
    response: Response,
    request: Request,
    token: str,
    *,
    secure: bool | None = None,
    samesite: str = "lax",
) -> None:
    settings = request.app.state.settings
    secure_cookie = _should_use_secure_cookie(request) if secure is None else secure
    response.set_cookie(
        key=settings.auth_cookie_name,
        value=token,
        httponly=True,
        secure=secure_cookie,
        samesite=samesite,
        max_age=settings.auth_session_duration_days * 24 * 60 * 60,
        path="/",
    )
    logger.debug(
        "auth cookie set cookie_name=%s secure=%s samesite=%s token_prefix=%s",
        settings.auth_cookie_name,
        secure_cookie,
        samesite,
        token[:8],
    )


def clear_auth_cookie(response: Response, request: Request) -> None:
    response.delete_cookie(key=request.app.state.settings.auth_cookie_name, path="/")


def _should_use_secure_cookie(request: Request) -> bool:
    settings = request.app.state.settings
    if bool(getattr(settings, "debug", False)):
        return False

    forwarded_proto = (request.headers.get("x-forwarded-proto") or "").split(",")[0].strip().lower()
    if forwarded_proto == "https":
        return True

    # Only mark auth cookies as Secure when the effective request scheme is HTTPS.
    # Plain HTTP LAN access such as http://192.168.x.x:5000 must stay non-Secure
    # or browsers will refuse to send the cookie back on the frontend redirect.
    return request.url.scheme.lower() == "https"
