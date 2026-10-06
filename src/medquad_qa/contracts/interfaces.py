"""Adapter interfaces (structural Protocols) and shared error types.

Owner: lead. Implementations live in the owning teammate's package:
  Retriever                        -> medquad_qa.retrieval   (retrieval-engineer)
  Generator / BatchGenerator       -> medquad_qa.models      (model-engineer; factory models.load_generator)
  AnswerabilityPredictor           -> medquad_qa.models      (model-engineer)
  QAPipeline + ReadinessReporter   -> medquad_qa.rag         (retrieval-engineer; factory rag.factory.build_pipeline)
  PipelineObserver                 -> medquad_qa.observability (service-platform-engineer)
  consumers                        -> medquad_qa.api / evaluation
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from medquad_qa.contracts.answerability import PredictorOutput
from medquad_qa.contracts.qa import (
    AbstentionReason,
    Citation,
    ExperimentMode,
    GenerationParams,
    QARequest,
    QAResponse,
    RetrievalHit,
)


# --------------------------------------------------------------------------- errors
class MedQuADError(Exception):
    """Base class for all project errors."""


class ArtifactUnavailableError(MedQuADError):
    """A required model/index/corpus artifact is missing or failed to load.
    The API maps this to 503 and readiness=false for the affected component."""


class RetrieverUnavailableError(ArtifactUnavailableError):
    """Retriever backend (e.g. Qdrant) unreachable or index missing."""


class ModeUnavailableError(ArtifactUnavailableError):
    """One experiment mode cannot be served (e.g. LoRA adapter missing) while others can.
    The API maps this to 503 with error_code='mode_unavailable'."""

    def __init__(self, mode: ExperimentMode, detail: str = "") -> None:
        self.mode = mode
        super().__init__(f"mode '{mode}' unavailable" + (f": {detail}" if detail else ""))


class GenerationError(MedQuADError):
    """The generator failed to produce output (e.g. prompt exceeds max_input_tokens). API -> 502."""


class GenerationTimeoutError(GenerationError):
    """Generation exceeded its time budget (enforced inside the Generator). API -> 504."""


class ContractViolationError(MedQuADError):
    """Data does not satisfy a shared contract (e.g. corpus_version mismatch). API -> 500."""


# --------------------------------------------------------------------------- generation I/O
class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class GenerationResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    model_version: str = Field(description="'<hf_model_id>@<revision>' or '...+lora:<run_id>'.")
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    finish_reason: Literal["stop", "length"] = "stop"


# --------------------------------------------------------------------------- readiness
class ComponentStatus(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(description="e.g. corpus, retriever:bm25, retriever:dense, generator:base, answerability.")
    ok: bool
    required: bool
    degraded: bool = False
    detail: str | None = None
    version: str | None = None


# --------------------------------------------------------------------------- protocols
@runtime_checkable
class Retriever(Protocol):
    name: str
    corpus_version: str
    index_version: str | None

    def retrieve(self, query: str, top_k: int) -> list[RetrievalHit]:
        """Return up to ``top_k`` (<= MAX_TOP_K) hits, ranks 1..n, sorted by descending relevance.

        Implementations may over-fetch internally (e.g. chunk-level) and aggregate to records.
        Raises RetrieverUnavailableError if the backend/index is unavailable.
        Returns [] (never raises) for a valid query with no matches.
        Must treat ``query`` as data (no query-language injection).
        """
        ...


@runtime_checkable
class Generator(Protocol):
    model_version: str

    def generate(self, messages: list[ChatMessage], params: GenerationParams) -> GenerationResult:
        """Apply the model's chat template and generate.

        temperature == 0 means greedy decoding: repeatable on the same hardware and batch shape;
        bitwise determinism across devices/batch sizes is not guaranteed.
        Raises GenerationTimeoutError when the internal time budget is exceeded,
        GenerationError on other failures (incl. prompt > max_input_tokens; never silently truncates),
        ArtifactUnavailableError at load.
        """
        ...


@runtime_checkable
class BatchGenerator(Generator, Protocol):
    def generate_batch(self, batch: list[list[ChatMessage]], params: GenerationParams) -> list[GenerationResult]:
        """Offline-evaluation throughput path; same semantics as ``generate`` per item."""
        ...


@runtime_checkable
class AnswerabilityPredictor(Protocol):
    model_version: str
    threshold_version: str

    def predict(self, question: str, evidence: list[str]) -> PredictorOutput:
        """Score whether the evidence suffices to answer the question (non-clinical).
        Aggregation = max over (question, evidence_i); empty evidence -> score 0.0, label False."""
        ...


@runtime_checkable
class ReadinessReporter(Protocol):
    def readiness(self) -> list[ComponentStatus]:
        """Per-component status from manifests/handles, without running a query. Never raises."""
        ...

    def available_modes(self) -> frozenset[ExperimentMode]: ...

    def versions(self) -> dict[str, str | None]:
        """Keys: model_version, finetuned_model_version, corpus_version, index_version, prompt_version, ..."""
        ...


@runtime_checkable
class QAPipeline(Protocol):
    def answer(self, request: QARequest, request_id: str) -> QAResponse:
        """Run the experiment mode in ``request``.

        - All modes: personalized-advice safety rules run first (abstention_reason=personalized_medical_advice).
        - base/finetuned: no retrieval; citations=[]; retrieved_record_ids=[].
        - rag/finetuned_rag: retrieve -> answerability gate -> fit evidence to the generator's input budget
          (drop lowest-ranked hits whole; warning "evidence_truncated") -> generate -> validate citations
          (strip any label/ID not supplied) -> abstain if evidence inadequate.
        Raises ModeUnavailableError / ArtifactUnavailableError / GenerationTimeoutError / GenerationError.
        """
        ...


@runtime_checkable
class PipelineObserver(Protocol):
    """Optional hooks for metrics. Implementations must not log evidence/question text;
    the pipeline swallows observer exceptions."""

    def on_retrieval(self, mode: ExperimentMode, hits: list[RetrievalHit]) -> None: ...

    def on_gate(self, mode: ExperimentMode, output: PredictorOutput | None, gate: str) -> None: ...

    def on_citations(self, mode: ExperimentMode, valid: list[Citation], invalid_ids: list[str]) -> None: ...

    def on_abstention(self, mode: ExperimentMode, reason: AbstentionReason) -> None: ...
