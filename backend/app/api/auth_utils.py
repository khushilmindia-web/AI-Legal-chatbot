from __future__ import annotations

from fastapi import HTTPException, Request, Response

def get_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return authorization.strip()


def get_request_token(request: Request) -> str | None:
    authorization = request.headers.get("Authorization")
    token = get_bearer_token(authorization)
    if token:
        return token
    cookie_name = request.app.state.settings.auth_cookie_name
    return request.cookies.get(cookie_name)


def get_optional_current_user(request: Request):
    token = get_request_token(request)
    if not token:
        return None
    return request.app.state.session_store.get_user_by_token(token)


def require_current_user(request: Request):
    user = get_optional_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


def set_auth_cookie(response: Response, request: Request, token: str) -> None:
    settings = request.app.state.settings
    response.set_cookie(
        key=settings.auth_cookie_name,
        value=token,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=settings.auth_session_duration_days * 24 * 60 * 60,
        path="/",
    )


def clear_auth_cookie(response: Response, request: Request) -> None:
    response.delete_cookie(key=request.app.state.settings.auth_cookie_name, path="/")
