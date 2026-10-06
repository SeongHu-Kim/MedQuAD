"""Background pipeline build and readiness evaluation.

Owner: service-platform-engineer. The API process starts serving /health/live immediately; the
pipeline (models, indexes) is built in a worker thread. A failed build keeps the process alive with
/health/ready = 503 and the failure reason, so operators can inspect it instead of a crash loop.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any, Literal

from medquad_qa.contracts import EXPERIMENT_MODES, ComponentStatus, ExperimentMode, PipelineObserver
from medquad_qa.observability.metrics import ServiceMetrics

log = logging.getLogger(__name__)

PipelineProvider = Callable[[PipelineObserver | None], Any]
BuildState = Literal["building", "ready", "failed"]


def default_pipeline_provider(observer: PipelineObserver | None) -> Any:
    """Production provider: ``medquad_qa.rag.factory.build_pipeline`` (retrieval-engineer)."""
    from medquad_qa.rag.factory import build_pipeline  # imported lazily: heavy deps

    return build_pipeline(observer=observer)


class PipelineHolder:
    def __init__(self, provider: PipelineProvider, observer: PipelineObserver | None, metrics: ServiceMetrics) -> None:
        self._provider = provider
        self._observer = observer
        self._metrics = metrics
        self._lock = threading.Lock()
        self.state: BuildState = "building"
        self.pipeline: Any = None
        self.failure: str | None = None
        self.build_seconds: float | None = None
        self.built = threading.Event()
        metrics.set_pipeline_state("building")

    def build(self) -> None:
        """Run the provider once. Never raises; the outcome is recorded in ``state``."""
        start = time.perf_counter()
        try:
            pipeline = self._provider(self._observer)
        except BaseException as exc:  # includes ImportError while the rag package is not implemented
            with self._lock:
                self.state = "failed"
                # Type plus a bounded message: these come from our own factories, never from requests.
                self.failure = f"{type(exc).__name__}: {str(exc)[:300]}"
            log.error("pipeline_build_failed", extra={"error_type": type(exc).__name__})
            if not isinstance(exc, Exception):
                raise
        else:
            with self._lock:
                self.pipeline = pipeline
                self.state = "ready"
            log.info("pipeline_built")
        finally:
            self.build_seconds = time.perf_counter() - start
            self._metrics.pipeline_build_seconds.set(self.build_seconds)
            self._metrics.set_pipeline_state(self.state)
            self.built.set()
            self.refresh_artifact_metrics()

    # ------------------------------------------------------------------ readiness
    def components(self) -> list[ComponentStatus]:
        reporter = getattr(self.pipeline, "readiness", None)
        if reporter is None:
            return []
        try:
            return list(reporter())
        except Exception as exc:  # contract says never raises; be defensive anyway
            return [ComponentStatus(name="readiness", ok=False, required=True, detail=type(exc).__name__)]

    def available_modes(self) -> frozenset[ExperimentMode]:
        if self.pipeline is None:
            return frozenset()
        fn = getattr(self.pipeline, "available_modes", None)
        if fn is None:
            return frozenset(EXPERIMENT_MODES)
        try:
            return frozenset(fn())
        except Exception:
            return frozenset()

    def versions(self) -> dict[str, str | None]:
        fn = getattr(self.pipeline, "versions", None)
        if fn is None:
            return {}
        try:
            return dict(fn())
        except Exception:
            return {}

    def readiness(self) -> tuple[bool, dict[str, Any]]:
        """(ready, body). Ready iff built, every required component ok, and >= 1 mode servable."""
        body: dict[str, Any] = {"pipeline": self.state}
        if self.state == "building":
            return False, {**body, "status": "starting"}
        if self.state == "failed":
            return False, {**body, "status": "not_ready", "failure": self.failure}
        comps = self.components()
        modes = sorted(self.available_modes())
        required_ok = all(c.ok for c in comps if c.required)
        degraded = any(c.degraded or (not c.ok and not c.required) for c in comps)
        ready = required_ok and bool(modes)
        body.update(
            status="ready" if ready else "not_ready",
            degraded=degraded,
            available_modes=modes,
            components=[c.model_dump() for c in comps],
        )
        return ready, body

    def refresh_artifact_metrics(self) -> None:
        m = self._metrics
        m.artifact_info.clear()
        m.component_ready.clear()
        for name, version in self.versions().items():
            if version:
                m.artifact_info.labels(artifact=name, version=version).set(1.0)
        for c in self.components():
            m.component_ready.labels(component=c.name, required=str(c.required).lower()).set(1.0 if c.ok else 0.0)
        modes = self.available_modes()
        for mode in EXPERIMENT_MODES:
            m.mode_available.labels(mode=mode).set(1.0 if mode in modes else 0.0)
