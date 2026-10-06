"""PipelineObserver implementation that feeds Prometheus metrics.

Owner: service-platform-engineer. Only numbers and bounded labels are recorded: never question,
evidence or answer text. Per-request outcomes (abstentions, citation failures) are counted from the
QAResponse in the API to avoid double counting; the hooks here record signals that the response
does not carry (retrieval score distributions, gate scores).
"""

from __future__ import annotations

import logging

from medquad_qa.contracts import AbstentionReason, Citation, ExperimentMode, PredictorOutput, RetrievalHit
from medquad_qa.observability.metrics import ServiceMetrics

log = logging.getLogger(__name__)

_RETRIEVER_FAMILIES = ("bm25", "dense", "hybrid_rrf", "hybrid_rrf+ce", "fixture")
_GATES = frozenset({"predictor", "heuristic", "off"})


def retriever_label(name: str) -> str:
    """Bounded retriever label: '<family>:<index text mode>' for known families, else 'other'."""
    family, _, text_mode = name.partition(":")
    if family not in _RETRIEVER_FAMILIES:
        return "other"
    return f"{family}:{text_mode}" if text_mode in ("answer", "qa") else family


class MetricsObserver:
    """Structural implementation of ``medquad_qa.contracts.PipelineObserver``."""

    def __init__(self, metrics: ServiceMetrics) -> None:
        self._m = metrics

    def on_retrieval(self, mode: ExperimentMode, hits: list[RetrievalHit]) -> None:
        try:
            self._m.retrieval_hits.labels(mode=mode).observe(len(hits))
            if hits:
                label = retriever_label(hits[0].retriever)
                self._m.retrieval_top_score.labels(retriever=label).observe(hits[0].score)
                for h in hits:
                    self._m.retrieval_score.labels(retriever=label).observe(h.score)
        except Exception:  # observers must never break the pipeline
            self._m.observer_errors.labels(hook="on_retrieval").inc()
            log.debug("observer on_retrieval failed", exc_info=True)

    def on_gate(self, mode: ExperimentMode, output: PredictorOutput | None, gate: str) -> None:
        try:
            g = gate if gate in _GATES else "other"
            if output is not None:
                self._m.answerability_score.labels(gate=g).observe(output.answerability_score)
                self._m.gate_decisions.labels(gate=g, label=str(output.predicted_label).lower()).inc()
            else:
                self._m.gate_decisions.labels(gate=g, label="none").inc()
        except Exception:
            self._m.observer_errors.labels(hook="on_gate").inc()
            log.debug("observer on_gate failed", exc_info=True)

    def on_citations(self, mode: ExperimentMode, valid: list[Citation], invalid_ids: list[str]) -> None:
        """Counted from the QAResponse in the API (see module docstring)."""

    def on_abstention(self, mode: ExperimentMode, reason: AbstentionReason) -> None:
        """Counted from the QAResponse in the API (see module docstring)."""
