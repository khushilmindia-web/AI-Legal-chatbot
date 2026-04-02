from __future__ import annotations

import logging


def configure_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s [request_id=%(request_id)s]: %(message)s",
    )

    class RequestContextFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            if not hasattr(record, "request_id"):
                record.request_id = "-"
            return True

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.addFilter(RequestContextFilter())
