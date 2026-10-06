"""``build_pipeline``: wire the configured retriever, generators and answerability gate.

Never raises for missing artifacts; unavailable components show up in ``readiness()`` and requests for an
unavailable mode raise ModeUnavailableError / ArtifactUnavailableError (API -> 503).
"""

from __future__ import annotations

import logging
from typing import Any

from medquad_qa.contracts import (
    AnswerabilityPredictor,
    ArtifactUnavailableError,
    ComponentStatus,
    Generator,
    PipelineObserver,
)
from medquad_qa.rag.gate import Gate, HeuristicGate
from medquad_qa.rag.pipeline import GeneratorProvider, RagPipeline
from medquad_qa.rag.settings import RagSettings
from medquad_qa.retrieval.factory import build_retriever

log = logging.getLogger(__name__)


def _default_generator_provider() -> GeneratorProvider:
    def provider(variant: str) -> Generator:
        try:
            from medquad_qa.models import load_generator
        except ImportError as exc:
            raise ArtifactUnavailableError("medquad_qa.models.load_generator not available") from exc
        gen: Generator = load_generator(variant)
        return gen

    return provider


def _default_generator_status() -> Any:
    try:
        from medquad_qa.models import generator_status
    except ImportError:
        return None
    return lambda variant: generator_status(variant)


def _load_predictor() -> tuple[AnswerabilityPredictor | None, str | None]:
    try:
        from medquad_qa.models import load_answerability_predictor  # type: ignore[attr-defined]
    except ImportError:
        return None, "predictor factory not available"
    try:
        predictor: AnswerabilityPredictor = load_answerability_predictor()
        return predictor, None
    except ArtifactUnavailableError as exc:
        return None, str(exc)
    except Exception as exc:
        return None, type(exc).__name__


def build_pipeline(
    settings: RagSettings | None = None,
    *,
    observer: PipelineObserver | None = None,
    generator_provider: GeneratorProvider | None = None,
    predictor: AnswerabilityPredictor | None = None,
    retriever: Any | None = None,
) -> RagPipeline:
    """Build the pipeline from settings (``RagSettings.from_env()`` when None).

    Keyword overrides exist for tests and evaluation (e.g. a FixtureRetriever or FakeGenerator).
    """
    s = settings or RagSettings.from_env()
    statuses: list[ComponentStatus] = []
    if retriever is None:
        bundle = build_retriever(s.retrieval)
        retriever = bundle.retriever
        statuses.extend(bundle.statuses)

    predictor_detail = None
    if predictor is None and s.gate_mode in ("auto", "predictor"):
        predictor, predictor_detail = _load_predictor()
    heuristic = HeuristicGate.from_file(s.gate_heuristic_path)
    mode = s.gate_mode
    if mode == "predictor" and predictor is None:
        log.warning("answerability predictor unavailable; falling back to heuristic gate")
        mode = "heuristic"
    gate = Gate(predictor, heuristic, mode)
    statuses.append(
        ComponentStatus(
            name="answerability",
            ok=True,
            required=False,
            degraded=gate.mode != "predictor" and s.gate_mode != "off",
            detail=f"gate={gate.mode}" + (f"; predictor: {predictor_detail}" if predictor_detail else ""),
            version=gate.threshold_version,
        )
    )
    pipeline = RagPipeline(
        retriever=retriever,
        generator_provider=generator_provider or _default_generator_provider(),
        gate=gate,
        observer=observer,
        max_input_tokens=s.max_input_tokens,
        component_statuses=statuses,
        generator_status=_default_generator_status() if generator_provider is None else None,
    )
    if s.preload_generators:
        pipeline.preload(s.preload_generators)
    return pipeline
