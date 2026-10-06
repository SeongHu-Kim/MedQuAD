"""Retrieval run file format (D-024): one JSON object per frozen query, IDs only (no question/evidence text).

Owner: evaluation-safety-engineer. Produced by ``medquad_qa.retrieval`` ``run``; consumed by the metric harness.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RunHit(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str
    chunk_id: str | None = None
    rank: int = Field(ge=1)
    score: float


class RetrievalRunRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    example_id: str
    retriever: str
    index_version: str | None = None
    corpus_version: str
    top_k: int = Field(ge=1)
    latency_ms: float = Field(ge=0.0)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    hits: list[RunHit] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_hits(self) -> RetrievalRunRecord:
        if len(self.hits) > self.top_k:
            raise ValueError(f"{self.example_id}: {len(self.hits)} hits exceed top_k={self.top_k}")
        if [h.rank for h in self.hits] != list(range(1, len(self.hits) + 1)):
            raise ValueError(f"{self.example_id}: ranks must be 1..n in order")
        ids = [h.record_id for h in self.hits]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{self.example_id}: duplicate record_id within one query")
        if self.error is not None and self.hits:
            raise ValueError(f"{self.example_id}: error rows must have no hits")
        return self

    def ranked_ids(self) -> list[str]:
        return [h.record_id for h in self.hits]


def load_run(path: str | Path, expected_example_ids: Iterable[str] | None = None) -> list[RetrievalRunRecord]:
    """Read and validate a run file. If ``expected_example_ids`` is given, the file must cover exactly that set."""
    rows: list[RetrievalRunRecord] = []
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rows.append(RetrievalRunRecord.model_validate(json.loads(line)))
            except ValueError as exc:  # json.JSONDecodeError and pydantic ValidationError are ValueErrors
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    seen = [r.example_id for r in rows]
    if len(set(seen)) != len(seen):
        raise ValueError(f"{path}: duplicate example_id rows")
    if expected_example_ids is not None:
        expected = set(expected_example_ids)
        missing, extra = expected - set(seen), set(seen) - expected
        if missing or extra:
            raise ValueError(f"{path}: example_id mismatch (missing={len(missing)}, unexpected={len(extra)})")
    return rows
