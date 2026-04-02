from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api.auth_utils import get_optional_current_user
from backend.app.api.routes.auth import router as auth_router
from backend.app.api.routes.chat import router as chat_router
from backend.app.api.routes.debug import router as debug_router
from backend.app.api.routes.health import router as health_router
from backend.app.core.config import get_settings
from backend.app.core.logging import configure_logging
from backend.app.services.session_store import SessionStore
from backend.app.utils.request_context import RequestContextMiddleware


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.debug)

    app = FastAPI(title="Lawyer AI", version="0.1.0")
    app.state.settings = settings
    app.state.session_store = SessionStore(settings.database_url)

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
            "/frontend/index.html",
            "/frontend/index.html".lower(),
        }
        path = request.url.path
        lower_path = path.lower()

        if path == "/":
            destination = settings.frontend_app_path if get_optional_current_user(request) else settings.frontend_auth_path
            return RedirectResponse(url=destination)

        if lower_path in {"/frontend", "/frontend/"}:
            destination = settings.frontend_app_path if get_optional_current_user(request) else settings.frontend_auth_path
            return RedirectResponse(url=destination)

        if path == "/frontend/Index.html":
            return RedirectResponse(url=settings.frontend_app_path)

        if lower_path in protected_paths and not get_optional_current_user(request):
            return RedirectResponse(url=settings.frontend_auth_path)

        return await call_next(request)

    if settings.frontend_dir.exists():
        app.mount("/frontend", StaticFiles(directory=str(settings.frontend_dir)), name="frontend")

    app.include_router(auth_router)
    app.include_router(health_router)
    app.include_router(debug_router)
    app.include_router(chat_router)
    return app


app = create_app()
