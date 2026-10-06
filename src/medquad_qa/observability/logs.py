"""Structured JSON logging without raw medical text by default.

Owner: service-platform-engineer. Log records carry request metadata (request_id, route, status,
mode, outcome, latency, counts, versions). Question and answer text are NOT logged unless
``MEDQUAD_LOG_CONTENT=true``; even then only a sha256 and the first 200 characters are kept
(see docs/operations/runbook.md, "Content logging").
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

CONTENT_PREVIEW_CHARS = 200

# Attributes every LogRecord has; anything else passed via ``extra=`` is emitted as a JSON field.
_STD_ATTRS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()) | {
    "message",
    "asctime",
    "color_message",  # uvicorn ANSI duplicate of msg
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _STD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info and record.exc_info[0] is not None:
            # Exception type only: messages/tracebacks could embed request content.
            payload["exc_type"] = record.exc_info[0].__name__
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    """Route the root logger (and uvicorn's loggers) to one JSON stdout handler."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers[:] = []
        lg.propagate = True


def content_fields(prefix: str, text: str) -> dict[str, str]:
    """Minimised content for opt-in logging: sha256 plus a 200-char preview."""
    return {
        f"{prefix}_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        f"{prefix}_preview": text[:CONTENT_PREVIEW_CHARS],
    }
