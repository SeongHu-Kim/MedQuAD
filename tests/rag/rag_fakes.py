"""RAG test doubles (imported by tests in this directory). All data SYNTHETIC."""

from __future__ import annotations

from collections.abc import Callable

from medquad_qa.contracts import (
    ArtifactUnavailableError,
    ChatMessage,
    GenerationParams,
    GenerationResult,
    PredictorOutput,
)
from medquad_qa.rag.gate import Gate, HeuristicGate
from medquad_qa.rag.pipeline import RagPipeline


class FakeGenerator:
    """Scripted generator: ``reply(messages) -> text``. Records every call."""

    def __init__(
        self,
        reply: str | Callable[[list[ChatMessage]], str] = "Answer [E1].",
        model_version: str = "fake/base@0",
        count_tokens: bool = False,
    ) -> None:
        self._reply = reply
        self.model_version = model_version
        self.calls: list[list[ChatMessage]] = []
        if count_tokens:
            self.count_tokens = lambda msgs: sum(len(m.content.split()) for m in msgs)  # type: ignore[method-assign]

    def generate(self, messages: list[ChatMessage], params: GenerationParams) -> GenerationResult:
        self.calls.append(messages)
        text = self._reply(messages) if callable(self._reply) else self._reply
        return GenerationResult(
            text=text,
            model_version=self.model_version,
            prompt_tokens=11,
            completion_tokens=7,
            latency_ms=1.0,
            finish_reason="stop",
        )


class FakePredictor:
    model_version = "fake-predictor"
    threshold_version = "fake-thr"

    def __init__(self, label: bool = True, raises: bool = False) -> None:
        self.label = label
        self.raises = raises

    def predict(self, question: str, evidence: list[str]) -> PredictorOutput:
        if self.raises:
            raise RuntimeError("boom")
        return PredictorOutput(
            answerability_score=0.9 if self.label else 0.1,
            predicted_label=self.label,
            threshold_version=self.threshold_version,
            model_version=self.model_version,
            per_evidence_scores=[0.5] * len(evidence),
        )


def make_pipeline(
    retriever, base: FakeGenerator | None = None, finetuned: FakeGenerator | None = None, gate: Gate | None = None, **kw
) -> RagPipeline:
    gens = {"base": base or FakeGenerator(), "finetuned": finetuned}

    def provider(variant: str):
        g = gens.get(variant)
        if g is None:
            raise ArtifactUnavailableError(f"{variant} missing")
        return g

    p = RagPipeline(
        retriever=retriever, generator_provider=provider, gate=gate or Gate(None, HeuristicGate(), "heuristic"), **kw
    )
    p.preload()
    return p
