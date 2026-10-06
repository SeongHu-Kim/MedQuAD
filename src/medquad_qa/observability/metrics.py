"""Prometheus metric catalogue (prefix ``medquad_``).

Owner: service-platform-engineer. Every metric lives in a per-app ``CollectorRegistry`` so tests and
multiple app instances never collide. Label values are bounded: modes, reasons and components come
from contract literals or allowlists; free-form values fall back to ``other``.

Nothing here measures live answer accuracy: there are no labels at serving time. Quality numbers
come only from offline evaluation runs (``medquad_offline_eval_metric``), with their run id.
"""

from __future__ import annotations

from typing import get_args

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Info

from medquad_qa.contracts import CONTRACTS_VERSION, AbstentionReason, QAResponse

LATENCY_BUCKETS_S = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0, 90.0, 120.0)
# Retrieval scores are retriever-specific and uncalibrated (RRF ~0.01-0.05, cosine 0-1, BM25 0-50).
SCORE_BUCKETS = (
    0.005,
    0.01,
    0.02,
    0.03,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    50.0,
)
PROB_BUCKETS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)
CHAR_BUCKETS = (10, 25, 50, 100, 200, 400, 800, 1200, 1600, 2000)
TOKEN_BUCKETS = (16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096)
COUNT_BUCKETS = (0, 1, 2, 3, 5, 10, 20)

COMPONENTS = (
    "safety_rules",
    "retrieval",
    "answerability_gate",
    "prompt_build",
    "generation",
    "citation_validation",
)
KNOWN_WARNINGS = frozenset({"lexical_fallback", "evidence_truncated"})
ABSTENTION_REASONS = frozenset(get_args(AbstentionReason))


def bounded(value: str | None, allowed: frozenset[str] | tuple[str, ...]) -> str:
    """Map a free-form label value to itself if allowlisted, else ``other``."""
    return value if value is not None and value in allowed else "other"


class ServiceMetrics:
    """All service metrics, registered on one registry."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry(auto_describe=True)
        r = self.registry

        # HTTP transport
        self.http_requests = Counter(
            "medquad_http_requests",
            "HTTP requests by route template and status.",
            ["method", "route", "status"],
            registry=r,
        )
        self.http_duration = Histogram(
            "medquad_http_request_duration_seconds",
            "HTTP request duration.",
            ["method", "route"],
            buckets=LATENCY_BUCKETS_S,
            registry=r,
        )
        self.http_inflight = Gauge("medquad_http_requests_inflight", "HTTP requests in progress.", registry=r)

        # QA outcomes (source of truth: the QAResponse, or the mapped error)
        self.qa_requests = Counter(
            "medquad_qa_requests",
            "QA requests by mode and outcome (answered|abstained|error).",
            ["mode", "outcome"],
            registry=r,
        )
        self.qa_errors = Counter(
            "medquad_qa_errors", "QA errors by mode and error code.", ["mode", "error_code"], registry=r
        )
        self.qa_latency = Histogram(
            "medquad_qa_latency_seconds",
            "End-to-end pipeline latency (successful responses).",
            ["mode"],
            buckets=LATENCY_BUCKETS_S,
            registry=r,
        )
        self.qa_component_latency = Histogram(
            "medquad_qa_component_latency_seconds",
            "Per-component pipeline latency.",
            ["mode", "component"],
            buckets=LATENCY_BUCKETS_S,
            registry=r,
        )
        self.qa_inflight = Gauge("medquad_qa_inflight", "QA requests holding a generation slot.", registry=r)
        self.abstentions = Counter(
            "medquad_qa_abstentions", "Abstentions by mode and reason.", ["mode", "reason"], registry=r
        )
        self.invalid_citation_ids = Counter(
            "medquad_qa_invalid_citation_ids",
            "Citation tokens stripped because they matched no supplied evidence.",
            ["mode"],
            registry=r,
        )
        self.citation_failures = Counter(
            "medquad_qa_citation_failures",
            "Responses with a citation-validation failure (kind: invalid_ids|invalid_citations|missing_citations).",
            ["mode", "kind"],
            registry=r,
        )
        self.citations = Histogram(
            "medquad_qa_citations_per_response",
            "Valid citations per response.",
            ["mode"],
            buckets=COUNT_BUCKETS,
            registry=r,
        )
        self.warnings = Counter("medquad_qa_warnings", "Pipeline warnings.", ["mode", "warning"], registry=r)
        self.input_chars = Histogram(
            "medquad_qa_input_chars",
            "Question length in characters (after stripping).",
            ["mode"],
            buckets=CHAR_BUCKETS,
            registry=r,
        )
        self.prompt_tokens = Histogram(
            "medquad_generation_prompt_tokens",
            "Prompt tokens per generation.",
            ["mode"],
            buckets=TOKEN_BUCKETS,
            registry=r,
        )
        self.completion_tokens = Histogram(
            "medquad_generation_completion_tokens",
            "Completion tokens per generation.",
            ["mode"],
            buckets=TOKEN_BUCKETS,
            registry=r,
        )
        self.finish_reason = Counter(
            "medquad_generation_finish",
            "Generation finish reasons (length = hit max_new_tokens).",
            ["mode", "reason"],
            registry=r,
        )

        # Pipeline-internal signals (from PipelineObserver; not present in QAResponse)
        self.retrieval_top_score = Histogram(
            "medquad_retrieval_top_score",
            "Top-1 retrieval score (uncalibrated; not medical confidence).",
            ["retriever"],
            buckets=SCORE_BUCKETS,
            registry=r,
        )
        self.retrieval_score = Histogram(
            "medquad_retrieval_score",
            "All returned retrieval scores (uncalibrated).",
            ["retriever"],
            buckets=SCORE_BUCKETS,
            registry=r,
        )
        self.retrieval_hits = Histogram(
            "medquad_retrieval_hits", "Hits returned per query.", ["mode"], buckets=COUNT_BUCKETS, registry=r
        )
        self.answerability_score = Histogram(
            "medquad_answerability_score",
            "Answerability score at the gate (non-clinical).",
            ["gate"],
            buckets=PROB_BUCKETS,
            registry=r,
        )
        self.gate_decisions = Counter(
            "medquad_answerability_gate_decisions", "Gate decisions.", ["gate", "label"], registry=r
        )
        self.observer_errors = Counter(
            "medquad_observer_errors",
            "Exceptions raised inside metric observer hooks (swallowed).",
            ["hook"],
            registry=r,
        )

        # Build / artifacts
        self.build_info = Info("medquad_build", "Service build information.", registry=r)
        self.build_info.info({"contracts_version": CONTRACTS_VERSION})
        self.pipeline_state = Gauge("medquad_pipeline_state", "Pipeline build state (one-hot).", ["state"], registry=r)
        self.pipeline_build_seconds = Gauge(
            "medquad_pipeline_build_seconds", "Wall time of the last pipeline build.", registry=r
        )
        self.artifact_info = Gauge(
            "medquad_artifact_info",
            "Loaded artifact versions (value is always 1).",
            ["artifact", "version"],
            registry=r,
        )
        self.component_ready = Gauge(
            "medquad_component_ready",
            "Component readiness from manifests (1 ok, 0 not ok).",
            ["component", "required"],
            registry=r,
        )
        self.mode_available = Gauge(
            "medquad_mode_available", "Experiment mode servable (1) or not (0).", ["mode"], registry=r
        )

        # Offline evaluation (periodic, labelled runs; never live accuracy)
        self.offline_eval = Gauge(
            "medquad_offline_eval_metric",
            "Metric from the latest offline evaluation run (not live accuracy).",
            ["eval_run_id", "track", "mode", "metric"],
            registry=r,
        )
        self.offline_eval_timestamp = Gauge(
            "medquad_offline_eval_timestamp_seconds",
            "Completion time of the exported offline evaluation run.",
            ["eval_run_id"],
            registry=r,
        )

    # ------------------------------------------------------------------ helpers
    def set_pipeline_state(self, state: str) -> None:
        for s in ("building", "ready", "failed"):
            self.pipeline_state.labels(state=s).set(1.0 if s == state else 0.0)

    def observe_response(self, resp: QAResponse, question_chars: int) -> None:
        """Record per-request outcome metrics from the (authoritative) pipeline response."""
        mode = resp.experiment_mode
        self.qa_requests.labels(mode=mode, outcome="abstained" if resp.abstained else "answered").inc()
        self.qa_latency.labels(mode=mode).observe(resp.latency_ms / 1000.0)
        self.input_chars.labels(mode=mode).observe(question_chars)
        for comp, ms in resp.component_latency_ms.items():
            self.qa_component_latency.labels(mode=mode, component=bounded(comp, COMPONENTS)).observe(ms / 1000.0)
        if resp.abstained:
            self.abstentions.labels(mode=mode, reason=bounded(resp.abstention_reason, ABSTENTION_REASONS)).inc()
            if resp.abstention_reason in ("invalid_citations", "missing_citations"):
                self.citation_failures.labels(mode=mode, kind=resp.abstention_reason).inc()
        if resp.invalid_citation_ids:
            self.invalid_citation_ids.labels(mode=mode).inc(len(resp.invalid_citation_ids))
            self.citation_failures.labels(mode=mode, kind="invalid_ids").inc()
        if not resp.abstained:
            self.citations.labels(mode=mode).observe(len(resp.citations))
        for w in resp.warnings:
            self.warnings.labels(mode=mode, warning=bounded(w, KNOWN_WARNINGS)).inc()
        if resp.generation is not None:
            self.prompt_tokens.labels(mode=mode).observe(resp.generation.prompt_tokens)
            self.completion_tokens.labels(mode=mode).observe(resp.generation.completion_tokens)
            self.finish_reason.labels(mode=mode, reason=resp.generation.finish_reason).inc()

    def observe_error(self, mode: str, error_code: str) -> None:
        self.qa_requests.labels(mode=mode, outcome="error").inc()
        self.qa_errors.labels(mode=mode, error_code=error_code).inc()
