"""Retrieval and question-answering contracts shared by retrieval, RAG, models and API.

Owner: lead.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from medquad_qa.contracts.answerability import PredictorOutput

ExperimentMode = Literal["base", "rag", "finetuned", "finetuned_rag"]
EXPERIMENT_MODES: tuple[ExperimentMode, ...] = ("base", "rag", "finetuned", "finetuned_rag")
RAG_MODES: frozenset[str] = frozenset({"rag", "finetuned_rag"})
FINETUNED_MODES: frozenset[str] = frozenset({"finetuned", "finetuned_rag"})

MAX_QUESTION_CHARS = 2000
MAX_TOP_K = 20
MAX_NEW_TOKENS = 1024

# request_id: opaque, <=64 chars of [A-Za-z0-9._-]; the API sanitises incoming X-Request-ID to this.
REQUEST_ID_PATTERN = r"^[A-Za-z0-9._-]{1,64}$"

# Disallowed: C0 control chars except \t \n \r, DEL, and C1 control chars.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


class RetrievalHit(BaseModel):
    """One retrieved evidence record. ``score`` is retriever-specific and uncalibrated:
    never present it as medical confidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str
    rank: int = Field(ge=1, description="1-based rank within this result list.")
    score: float
    retriever: str = Field(description="Includes index text mode, e.g. 'bm25:answer', 'hybrid_rrf:qa'.")
    evidence_text: str
    chunk_id: str | None = Field(default=None, description="'<record_id>#c<i>' when chunked.")
    evidence_char_start: int | None = Field(default=None, ge=0, description="Offset into MedicalRecord.answer.")
    evidence_char_end: int | None = Field(default=None, ge=0)
    record_question: str | None = None
    source_name: str | None = None
    source_url: str | None = None
    topic: str | None = None
    duplicate_group_id: str | None = None
    split_group_id: str | None = None
    quality_flags: list[str] = Field(default_factory=list, description="Copied from MedicalRecord.quality_flags.")
    corpus_version: str
    index_version: str | None = None


class GenerationParams(BaseModel):
    """Caller-adjustable generation settings, bounded to safe ranges."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_new_tokens: int = Field(default=256, ge=1, le=MAX_NEW_TOKENS)
    temperature: float = Field(default=0.0, ge=0.0, le=1.5, description="0.0 means greedy decoding.")
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int | None = Field(default=None, ge=0)


class QARequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str = Field(min_length=3, max_length=MAX_QUESTION_CHARS)
    experiment_mode: ExperimentMode = "rag"
    top_k: int = Field(default=5, ge=1, le=MAX_TOP_K)
    generation: GenerationParams | None = None

    @field_validator("question", mode="before")
    @classmethod
    def _clean_question(cls, v: object) -> object:
        """Strip surrounding whitespace before length checks; reject control characters."""
        if isinstance(v, str):
            if _CONTROL_CHARS.search(v):
                raise ValueError("question contains control characters")
            return v.strip()
        return v


class Citation(BaseModel):
    """A citation to a record that was actually retrieved AND supplied to the generator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str
    chunk_id: str | None = None
    source_name: str | None = None
    source_url: str | None = Field(default=None, description="Only if present in the corpus; never fabricated.")
    topic: str | None = None
    evidence_snippet: str = Field(description="Excerpt of the supplied evidence text.")


AbstentionReason = Literal[
    "no_relevant_evidence",
    "insufficient_evidence",
    "personalized_medical_advice",
    "invalid_citations",  # only invalid citations were emitted
    "missing_citations",  # no citation at all in a RAG answer
    "out_of_scope",
    "generation_failed",
]


class GenerationMeta(BaseModel):
    """Effective generation settings and token accounting, for matched-budget evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    params: GenerationParams
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    finish_reason: Literal["stop", "length"]


class QAResponse(BaseModel):
    """Pipeline output.

    Citation format: the model cites per-request labels [E1]..[Ek]; the pipeline maps them to
    record IDs and rewrites the answer text to inline ``[mq-...]`` markers. Any label/ID not in the
    supplied evidence is stripped and listed in ``invalid_citation_ids``.
    Invariant: {c.record_id for c in citations} <= set(retrieved_record_ids).

    ``retrieved_record_ids`` is in rank order. When generation ran, it lists exactly the records
    supplied to the generator; when the pipeline abstained before generation, it lists the retrieved
    records for transparency and ``citations`` is empty.
    ``model_version`` is always the configured generator's version, even when abstaining early.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str
    experiment_mode: ExperimentMode
    answer: str = Field(description="Answer text, or a safe refusal/abstention message when abstained.")
    abstained: bool
    abstention_reason: AbstentionReason | None = None
    citations: list[Citation] = Field(default_factory=list)
    retrieved_record_ids: list[str] = Field(default_factory=list)
    invalid_citation_ids: list[str] = Field(
        default_factory=list, description="Raw citation tokens not matching supplied evidence (stripped)."
    )
    retriever: str | None = Field(default=None, description="Retriever actually used, incl. fallback.")
    answerability: PredictorOutput | None = None
    answerability_gate: Literal["predictor", "heuristic", "off"] | None = None
    generation: GenerationMeta | None = Field(default=None, description="None when generation did not run.")
    warnings: list[str] = Field(default_factory=list, description="e.g. 'lexical_fallback'.")
    model_version: str
    corpus_version: str | None = None
    index_version: str | None = None
    prompt_version: str
    latency_ms: float = Field(ge=0.0)
    component_latency_ms: dict[str, float] = Field(
        default_factory=dict,
        description="Keys: safety_rules, retrieval, answerability_gate, prompt_build, generation, citation_validation.",
    )
    disclaimer: str = (
        "Research prototype for general medical information only. Not medical advice; "
        "not clinically validated. Consult a qualified clinician."
    )
