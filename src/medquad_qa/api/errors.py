"""Error envelope and exception -> HTTP mapping (docs/architecture/contracts.md).

Owner: service-platform-engineer. Error bodies never echo request content or internal messages.
"""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from medquad_qa.contracts import (
    ArtifactUnavailableError,
    GenerationError,
    GenerationParams,
    GenerationTimeoutError,
    ModeUnavailableError,
    QARequest,
)

# loc elements that may be returned verbatim; anything else (e.g. a client-chosen extra key) is masked.
_KNOWN_LOC = (
    frozenset({"body", "query", "path", "header"}) | set(QARequest.model_fields) | set(GenerationParams.model_fields)
)


def _safe_loc(loc: tuple[Any, ...] | list[Any]) -> list[str]:
    return [str(p) if isinstance(p, int) or p in _KNOWN_LOC else "<extra>" for p in loc]


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    request_id: str
    errors: list[dict[str, Any]] | None = None


def request_id_of(request: Request) -> str:
    return str(getattr(request.state, "request_id", "unknown"))


def error_response(
    request: Request,
    status: int,
    error_code: str,
    message: str,
    *,
    retry_after: int | None = None,
    errors: list[dict[str, Any]] | None = None,
) -> JSONResponse:
    body = ErrorResponse(error_code=error_code, message=message, request_id=request_id_of(request), errors=errors)
    headers = {"Retry-After": str(retry_after)} if retry_after is not None else None
    return JSONResponse(body.model_dump(exclude_none=True), status_code=status, headers=headers)


def map_pipeline_error(exc: BaseException) -> tuple[int, str, str]:
    """(status, error_code, public message). Order matters: subclasses before parents."""
    if isinstance(exc, ModeUnavailableError):
        return 503, "mode_unavailable", f"Experiment mode '{exc.mode}' is not available."
    if isinstance(exc, ArtifactUnavailableError):
        return 503, "artifact_unavailable", "A required model or index artifact is unavailable."
    if isinstance(exc, GenerationTimeoutError):
        return 504, "generation_timeout", "Generation exceeded its time budget."
    if isinstance(exc, GenerationError):
        return 502, "generation_failed", "The generator failed to produce an answer."
    return 500, "internal_error", "Internal server error."


async def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """422 listing field locations and error types only: never the submitted input."""
    assert isinstance(exc, RequestValidationError)
    details = [
        {"loc": _safe_loc(err.get("loc", ())), "type": str(err.get("type", "value_error"))} for err in exc.errors()
    ]
    return error_response(request, 422, "validation_error", "Request validation failed.", errors=details)


async def http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    codes = {404: "not_found", 405: "method_not_allowed"}
    resp = error_response(request, exc.status_code, codes.get(exc.status_code, "http_error"), "Request failed.")
    if exc.headers:
        resp.headers.update(exc.headers)
    return resp
