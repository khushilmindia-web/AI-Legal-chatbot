from __future__ import annotations

import logging
import time
from contextvars import ContextVar
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        kwargs.setdefault("extra", {})
        kwargs["extra"]["request_id"] = request_id_var.get("-")
        return msg, kwargs


def get_logger(name: str) -> RequestIdAdapter:
    return RequestIdAdapter(logging.getLogger(name), {})


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid4())
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
            response.headers["x-request-id"] = request_id
            return response
        finally:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            logging.getLogger("lawyer_ai.request").info(
                "%s %s completed in %sms",
                request.method,
                request.url.path,
                elapsed_ms,
                extra={"request_id": request_id},
            )
            request_id_var.reset(token)
