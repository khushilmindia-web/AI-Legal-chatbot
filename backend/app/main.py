from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
import logging
import time
from datetime import datetime, timezone

from backend.app.api.auth_utils import get_optional_current_user
from backend.app.api.routes.admin import router as admin_router
from backend.app.api.routes.auth import router as auth_router
from backend.app.api.routes.chat import router as chat_router
from backend.app.api.routes.debug import router as debug_router
from backend.app.api.routes.health import router as health_router
from backend.app.core.config import get_settings
from backend.app.core.logging import configure_logging
from backend.app.services.chat_service import ChatService
from backend.app.services.storage import create_session_store
from backend.app.utils.request_context import RequestContextMiddleware

logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.debug)
    logger.info(
        "startup APP_BASE_URL=%s GOOGLE_REDIRECT_URI=%s effective_google_redirect_uri=%s CORS_ALLOW_ORIGINS=%s",
        settings.app_base_url,
        settings.google_redirect_uri,
        settings.google_oauth_redirect_uri,
        settings.cors_origins,
    )

    app = FastAPI(title="Lawyer AI", version="0.1.0")
    app.state.settings = settings
    app.state.started_at = datetime.now(timezone.utc)
    app.state.health_metrics = {"chat_response_times_ms": []}
    # SQLite temporarily disabled during MongoDB migration.
    # The old SQLite SessionStore code remains in backend/app/services/session_store.py
    # as a backup, but runtime storage is intentionally created through MongoDB only.
    app.state.session_store = create_session_store(settings)
    app.state.chat_service = ChatService(settings=settings, store=app.state.session_store)

    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def frontend_auth_guard(request, call_next):
        protected_paths = {
            settings.frontend_app_path.lower(),
            "/frontend/admin.html",
            "/frontend/index.html",
            "/frontend/index.html".lower(),
        }
        path = request.url.path
        lower_path = path.lower()

        if path == "/":
            user = get_optional_current_user(request)
            destination = settings.frontend_app_path if user else settings.frontend_auth_path
            logger.info(
                "frontend_auth_guard redirect path=%s destination=%s reason=%s cookies=%s",
                path,
                destination,
                "authenticated_root" if user else "unauthenticated_root",
                sorted(request.cookies.keys()),
            )
            return RedirectResponse(url=destination)

        if lower_path in {"/frontend", "/frontend/"}:
            user = get_optional_current_user(request)
            destination = settings.frontend_app_path if user else settings.frontend_auth_path
            logger.info(
                "frontend_auth_guard redirect path=%s destination=%s reason=%s cookies=%s",
                path,
                destination,
                "authenticated_frontend_root" if user else "unauthenticated_frontend_root",
                sorted(request.cookies.keys()),
            )
            return RedirectResponse(url=destination)

        if path == "/frontend/Index.html":
            logger.info("frontend_auth_guard redirect path=%s destination=%s reason=legacy_index_case", path, settings.frontend_app_path)
            return RedirectResponse(url=settings.frontend_app_path)

        if lower_path in protected_paths:
            user = get_optional_current_user(request)
            if not user:
                logger.warning(
                    "frontend_auth_guard redirect path=%s destination=%s reason=protected_path_no_session cookie_name=%s cookies=%s",
                    path,
                    settings.frontend_auth_path,
                    settings.auth_cookie_name,
                    sorted(request.cookies.keys()),
                )
                return RedirectResponse(url=settings.frontend_auth_path)
            logger.info(
                "frontend_auth_guard allow path=%s reason=protected_path_session user_id=%s cookie_name=%s cookies=%s",
                path,
                user.get("id"),
                settings.auth_cookie_name,
                sorted(request.cookies.keys()),
            )

        return await call_next(request)

    @app.middleware("http")
    async def chat_health_metrics(request, call_next):
        started = time.perf_counter()
        try:
            return await call_next(request)
        finally:
            if request.url.path in {"/chat", "/chat/upload"}:
                samples = app.state.health_metrics.setdefault("chat_response_times_ms", [])
                samples.append((time.perf_counter() - started) * 1000)
                del samples[:-100]

    if settings.frontend_dir.exists():
        app.mount("/frontend", StaticFiles(directory=str(settings.frontend_dir)), name="frontend")

    app.include_router(auth_router)
    app.include_router(health_router)
    app.include_router(debug_router)
    app.include_router(chat_router)
    app.include_router(admin_router)
    return app


app = create_app()
