"""Pure-ASGI middleware: request IDs, body-size limit, HTTP metrics and access logs.

Owner: service-platform-engineer.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from medquad_qa.contracts import REQUEST_ID_PATTERN
from medquad_qa.observability.metrics import ServiceMetrics

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

REQUEST_ID_HEADER = "x-request-id"
_REQUEST_ID_RE = re.compile(REQUEST_ID_PATTERN)
access_log = logging.getLogger("medquad_qa.api.access")


def sanitise_request_id(raw: str | None) -> str:
    """Keep a client X-Request-ID only if it matches REQUEST_ID_PATTERN; otherwise mint a new one."""
    if raw is not None and _REQUEST_ID_RE.fullmatch(raw):
        return raw
    return uuid.uuid4().hex


def _header(scope: Scope, name: bytes) -> str | None:
    for k, v in scope.get("headers", []):
        if k.lower() == name:
            try:
                return str(v.decode("latin-1"))
            except UnicodeDecodeError:  # pragma: no cover - latin-1 decodes any byte
                return None
    return None


async def _send_json(send: Send, status: int, body: dict[str, Any], request_id: str) -> None:
    data = json.dumps(body).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(data)).encode()),
                (REQUEST_ID_HEADER.encode(), request_id.encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": data})


class ServiceMiddleware:
    """Outermost app middleware.

    - Sanitises ``X-Request-ID`` (stored in ``scope["state"]["request_id"]``, echoed on every response).
    - Rejects bodies above ``max_body_bytes`` with 413 (Content-Length or streamed size), before parsing.
    - Records HTTP metrics by route template (unmatched paths -> ``unmatched``) and writes one access
      log line per request, without query strings or bodies.
    """

    def __init__(self, app: ASGIApp, *, metrics: ServiceMetrics, max_body_bytes: int) -> None:
        self.app = app
        self.metrics = metrics
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = sanitise_request_id(_header(scope, REQUEST_ID_HEADER.encode()))
        scope.setdefault("state", {})["request_id"] = request_id
        method = scope.get("method", "")
        start = time.perf_counter()
        status_holder = {"status": 500}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
                headers = [(k, v) for k, v in message.get("headers", []) if k.lower() != REQUEST_ID_HEADER.encode()]
                headers.append((REQUEST_ID_HEADER.encode(), request_id.encode()))
                message["headers"] = headers
            await send(message)

        self.metrics.http_inflight.inc()
        try:
            replay = await self._read_bounded_body(scope, receive)
            if replay is None:
                status_holder["status"] = 413
                await _send_json(
                    send,
                    413,
                    {
                        "error_code": "payload_too_large",
                        "message": f"Request body exceeds {self.max_body_bytes} bytes.",
                        "request_id": request_id,
                    },
                    request_id,
                )
                return
            await self.app(scope, replay, send_wrapper)
        finally:
            self.metrics.http_inflight.dec()
            elapsed = time.perf_counter() - start
            route = scope.get("route")
            route_label = getattr(route, "path", None) or "unmatched"
            status = status_holder["status"]
            self.metrics.http_requests.labels(method=method, route=route_label, status=str(status)).inc()
            self.metrics.http_duration.labels(method=method, route=route_label).observe(elapsed)
            access_log.info(
                "http_request",
                extra={
                    "request_id": request_id,
                    "method": method,
                    "route": route_label,
                    "status": status,
                    "duration_ms": round(elapsed * 1000.0, 2),
                },
            )

    async def _read_bounded_body(self, scope: Scope, receive: Receive) -> Receive | None:
        """Read the whole body (bounded) and return a replaying ``receive``; None if over the limit."""
        declared = _header(scope, b"content-length")
        if declared is not None:
            try:
                if int(declared) > self.max_body_bytes:
                    return None
            except ValueError:
                return None
        chunks: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                pending: list[Message] = [message]
                break
            if message["type"] != "http.request":  # pragma: no cover - not produced by servers
                continue
            body = message.get("body", b"")
            size += len(body)
            if size > self.max_body_bytes:
                return None
            chunks.append(body)
            if not message.get("more_body", False):
                pending = [{"type": "http.request", "body": b"".join(chunks), "more_body": False}]
                break

        async def replay() -> Message:
            if pending:
                return pending.pop(0)
            return await receive()

        return replay
