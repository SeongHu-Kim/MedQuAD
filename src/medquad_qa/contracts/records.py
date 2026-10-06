"""Corpus-level contracts: provenance and canonical medical records.

Owner: lead. Teammates may propose changes via the lead; never edit silently.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Provenance(BaseModel):
    """Where a record came from. Unknown values stay ``None``; never fabricate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_name: str = Field(description="Human-readable dataset name, e.g. 'MedQuAD (Kaggle CSV export)'.")
    source_file: str = Field(description="Path of the raw file relative to the repo root.")
    source_file_sha256: str = Field(min_length=64, max_length=64)
    row_index: int = Field(ge=0, description="0-based data-row index in the raw file (header excluded).")
    kaggle_dataset_ref: str | None = Field(default=None, description="owner/slug if verifiable, else None.")
    pipeline_version: str = Field(description="Version of the ingestion pipeline that produced the record.")
    duplicate_row_indices: list[int] = Field(
        default_factory=list, description="Row indices of excluded exact-duplicate copies collapsed into this record."
    )


class MedicalRecord(BaseModel):
    """One question/answer pair from the corpus.

    ``question``/``answer`` are the *normalized* forms used for indexing and
    training. ``question_raw``/``answer_raw`` preserve the original text exactly.
    Normalization must never drop measurements, negations or punctuation that
    carries meaning, and must not blindly lowercase.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str = Field(pattern=r"^mq-[0-9a-f]{16}$", description="'mq-' + 16 hex of a content hash (no row index).")
    question: str
    answer: str
    question_raw: str
    answer_raw: str
    source_url: str | None = None
    source_name: str | None = Field(default=None, description="Dataset 'source' column, e.g. 'GHR'.")
    source_document_id: str | None = None
    topic: str | None = Field(default=None, description="Dataset 'focus_area' column.")
    question_type: str | None = Field(
        default=None,
        description="Rule-derived from the question template (configs/data/question_types.yaml); not MedQuAD qtype.",
    )
    provenance: Provenance
    content_hash: str = Field(description="sha256 of NORMALIZED question + '\\x1f' + NORMALIZED answer.")
    duplicate_group_id: str = Field(description="Records sharing exact/near-duplicate content share this ID.")
    split_group_id: str = Field(description="Leakage-control unit: every record in a group lands in one split.")
    quality_flags: list[str] = Field(
        default_factory=list, description="e.g. boilerplate_answer, resource_list_answer, answer_starts_with_question."
    )
