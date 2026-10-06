"""FastAPI application factory.

Owner: service-platform-engineer. Endpoints: POST /v1/qa, GET /v1/info, GET /health/live,
GET /health/ready, GET /metrics. See docs/service/api.md.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.exceptions import HTTPException as StarletteHTTPException

from medquad_qa.api.errors import (
    ErrorResponse,
    error_response,
    http_exception_handler,
    map_pipeline_error,
    request_id_of,
    validation_exception_handler,
)
from medquad_qa.api.middleware import ServiceMiddleware
from medquad_qa.api.pipeline_state import PipelineHolder, PipelineProvider, default_pipeline_provider
from medquad_qa.api.settings import SERVICE_VERSION, ServiceSettings
from medquad_qa.contracts import (
    CONTRACTS_VERSION,
    MAX_NEW_TOKENS,
    MAX_QUESTION_CHARS,
    MAX_TOP_K,
    QARequest,
    QAResponse,
)
from medquad_qa.observability.logs import content_fields
from medquad_qa.observability.metrics import ServiceMetrics
from medquad_qa.observability.observer import MetricsObserver
from medquad_qa.observability.offline_eval import OfflineEvalExporter

log = logging.getLogger("medquad_qa.api")

_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse} for code in (413, 422, 500, 502, 503, 504)
}


class _Runtime:
    """Per-app mutable runtime objects (created in the lifespan, bound to its event loop)."""

    def __init__(self, settings: ServiceSettings) -> None:
        self.semaphore = asyncio.Semaphore(settings.max_concurrent_generations)
        self.executor = ThreadPoolExecutor(
            max_workers=settings.max_concurrent_generations, thread_name_prefix="medquad-gen"
        )


def create_app(settings: ServiceSettings | None = None, pipeline_provider: PipelineProvider | None = None) -> FastAPI:
    settings = settings or ServiceSettings()
    metrics = ServiceMetrics()
    metrics.build_info.info({"contracts_version": CONTRACTS_VERSION, "service_version": SERVICE_VERSION})
    observer = MetricsObserver(metrics)
    holder = PipelineHolder(pipeline_provider or default_pipeline_provider, observer, metrics)
    offline_eval = OfflineEvalExporter(metrics, settings.offline_eval_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if settings.log_content:
            log.warning(
                "content_logging_enabled",
                extra={"detail": "MEDQUAD_LOG_CONTENT=true: question/answer sha256 + 200-char previews are logged."},
            )
        runtime = _Runtime(settings)
        app.state.runtime = runtime
        build_task: asyncio.Future[None] | None = None
        if settings.build_in_background:
            build_task = asyncio.get_running_loop().run_in_executor(None, holder.build)
        else:
            holder.build()
        try:
            yield
        finally:
            if build_task is not None and not build_task.done():
                log.warning("shutdown_during_pipeline_build")
            runtime.executor.shutdown(wait=False, cancel_futures=True)
            close = getattr(holder.pipeline, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    log.exception("pipeline_close_failed")

    app = FastAPI(
        title="MedQuAD Evidence-Grounded QA",
        version=SERVICE_VERSION,
        description="Research prototype for general medical information only. Not medical advice; "
        "not clinically validated.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.metrics = metrics
    app.state.pipeline = holder
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_middleware(ServiceMiddleware, metrics=metrics, max_body_bytes=settings.max_body_bytes)

    # ------------------------------------------------------------------ health
    @app.get("/health/live", tags=["health"])
    async def live() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready", tags=["health"], responses={503: {"description": "Not ready"}})
    async def ready() -> JSONResponse:
        ok, body = holder.readiness()
        headers = None if ok else {"Retry-After": str(settings.retry_after_s)}
        return JSONResponse(body, status_code=200 if ok else 503, headers=headers)

    @app.get("/metrics", tags=["ops"], include_in_schema=False)
    async def metrics_endpoint() -> Response:
        offline_eval.refresh()
        if holder.state == "ready":
            holder.refresh_artifact_metrics()
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/info", tags=["qa"])
    async def info() -> dict[str, Any]:
        return {
            "service_version": SERVICE_VERSION,
            "contracts_version": CONTRACTS_VERSION,
            "pipeline": holder.state,
            "available_modes": sorted(holder.available_modes()),
            "versions": holder.versions(),
            "limits": {
                "max_question_chars": MAX_QUESTION_CHARS,
                "max_top_k": MAX_TOP_K,
                "max_new_tokens": MAX_NEW_TOKENS,
                "max_body_bytes": settings.max_body_bytes,
                "request_timeout_s": settings.request_timeout_s,
            },
            "disclaimer": QAResponse.model_fields["disclaimer"].default,
        }

    # ------------------------------------------------------------------ QA
    @app.post("/v1/qa", response_model=QAResponse, tags=["qa"], responses=_ERROR_RESPONSES)
    async def qa(body: QARequest, request: Request) -> Any:
        request_id = request_id_of(request)
        mode = body.experiment_mode
        retry = settings.retry_after_s

        def fail(status: int, code: str, message: str, *, retry_after: int | None = None) -> JSONResponse:
            metrics.observe_error(mode, code)
            log.info(
                "qa_request", extra={"request_id": request_id, "mode": mode, "outcome": "error", "error_code": code}
            )
            return error_response(request, status, code, message, retry_after=retry_after)

        if holder.state != "ready":
            return fail(503, "not_ready", "The QA pipeline is not ready.", retry_after=retry)
        if mode not in holder.available_modes():
            return fail(503, "mode_unavailable", f"Experiment mode '{mode}' is not available.", retry_after=retry)

        runtime: _Runtime = request.app.state.runtime
        try:
            await asyncio.wait_for(runtime.semaphore.acquire(), timeout=settings.queue_timeout_s)
        except TimeoutError:
            return fail(503, "busy", "All generation slots are busy; retry later.", retry_after=retry)

        metrics.qa_inflight.inc()
        start = time.perf_counter()
        loop = asyncio.get_running_loop()
        try:
            future = loop.run_in_executor(runtime.executor, holder.pipeline.answer, body, request_id)
        except Exception:
            runtime.semaphore.release()
            metrics.qa_inflight.dec()
            raise

        def _release(_: object) -> None:
            # The slot is freed only when the worker thread really finishes, even after a 504:
            # a timed-out generation still occupies the GPU until it returns.
            loop.call_soon_threadsafe(runtime.semaphore.release)
            loop.call_soon_threadsafe(metrics.qa_inflight.dec)

        future.add_done_callback(_release)
        try:
            resp: QAResponse = await asyncio.wait_for(asyncio.shield(future), timeout=settings.request_timeout_s)
        except TimeoutError:
            return fail(504, "request_timeout", "The request exceeded the service time budget.")
        except Exception as exc:
            status, code, message = map_pipeline_error(exc)
            if status == 500:
                log.error("qa_internal_error", extra={"request_id": request_id, "error_type": type(exc).__name__})
            return fail(status, code, message, retry_after=retry if status == 503 else None)

        metrics.observe_response(resp, len(body.question))
        fields: dict[str, Any] = {
            "request_id": request_id,
            "mode": mode,
            "outcome": "abstained" if resp.abstained else "answered",
            "abstention_reason": resp.abstention_reason,
            "top_k": body.top_k,
            "question_chars": len(body.question),
            "retrieved": len(resp.retrieved_record_ids),
            "citations": len(resp.citations),
            "invalid_citations": len(resp.invalid_citation_ids),
            "warnings": resp.warnings,
            "pipeline_latency_ms": round(resp.latency_ms, 2),
            "wall_ms": round((time.perf_counter() - start) * 1000.0, 2),
            "model_version": resp.model_version,
            "index_version": resp.index_version,
            "prompt_version": resp.prompt_version,
        }
        if settings.log_content:
            fields.update(content_fields("question", body.question))
            fields.update(content_fields("answer", resp.answer))
        log.info("qa_request", extra=fields)
        if resp.request_id != request_id:
            resp = resp.model_copy(update={"request_id": request_id})
        return resp

    return app
