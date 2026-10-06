"""In-memory corpus/chunk store shared by all retrievers, plus chunk -> record aggregation (MaxP)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from medquad_qa.contracts import ContractViolationError, MedicalRecord, RetrievalHit
from medquad_qa.retrieval.corpus import (
    Chunk,
    chunk_corpus,
    index_text,
    load_corpus,
    read_corpus_sha256,
    read_corpus_version,
    sha256_file,
)


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk_idx: int
    score: float


@dataclass(frozen=True, slots=True)
class RecordCandidate:
    """A record-level candidate after aggregation; ``chunk_idx`` is the best-scoring chunk (MaxP)."""

    record_id: str
    chunk_idx: int
    score: float


@dataclass
class CorpusStore:
    records: list[MedicalRecord]  # sorted by record_id
    corpus_version: str
    corpus_sha256: str | None
    max_words: int = 200
    overlap: int = 40
    chunks: list[Chunk] = field(init=False)
    by_id: dict[str, MedicalRecord] = field(init=False)
    chunk_index: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.records = sorted(self.records, key=lambda r: r.record_id)
        self.by_id = {r.record_id: r for r in self.records}
        self.chunks = chunk_corpus(self.records, self.max_words, self.overlap)
        self.chunk_index = {c.chunk_id: i for i, c in enumerate(self.chunks)}

    @classmethod
    def from_files(
        cls, corpus_path: Path, manifest_path: Path, *, max_words: int = 200, overlap: int = 40
    ) -> CorpusStore:
        records = load_corpus(corpus_path)
        digest = sha256_file(corpus_path)
        expected = read_corpus_sha256(manifest_path)
        if expected is not None and expected != digest:
            raise ContractViolationError(f"{corpus_path} sha256 does not match {manifest_path} (rebuild the corpus)")
        return cls(
            records=records,
            corpus_version=read_corpus_version(manifest_path, corpus_path),
            corpus_sha256=digest,
            max_words=max_words,
            overlap=overlap,
        )

    def index_texts(self, mode: str) -> list[str]:
        return [index_text(c, self.by_id[c.record_id], mode) for c in self.chunks]

    def chunk_ids(self) -> list[str]:
        return [c.chunk_id for c in self.chunks]

    def make_hit(self, cand: RecordCandidate, rank: int, retriever: str, index_version: str | None) -> RetrievalHit:
        chunk = self.chunks[cand.chunk_idx]
        rec = self.by_id[cand.record_id]
        return RetrievalHit(
            record_id=rec.record_id,
            rank=rank,
            score=float(cand.score),
            retriever=retriever,
            evidence_text=chunk.text,
            chunk_id=chunk.chunk_id,
            evidence_char_start=chunk.char_start,
            evidence_char_end=chunk.char_end,
            record_question=rec.question,
            source_name=rec.source_name,
            source_url=rec.source_url,
            topic=rec.topic,
            duplicate_group_id=rec.duplicate_group_id,
            split_group_id=rec.split_group_id,
            quality_flags=list(rec.quality_flags),
            corpus_version=self.corpus_version,
            index_version=index_version,
        )


def max_p(
    store: CorpusStore, scored: Iterable[ScoredChunk], limit: int, *, collapse_duplicates: bool = False
) -> list[RecordCandidate]:
    """Aggregate chunk scores to records by maximum (MaxP). ``scored`` must be sorted best-first
    with deterministic tie-breaking. Optionally keep only the best record per (duplicate_group_id, topic):
    duplicate groups can join templated answers about different conditions, so topic is part of the key."""
    out: list[RecordCandidate] = []
    seen_records: set[str] = set()
    seen_groups: set[tuple[str, str | None]] = set()
    for sc in scored:
        chunk = store.chunks[sc.chunk_idx]
        if chunk.record_id in seen_records:
            continue
        rec = store.by_id[chunk.record_id]
        group = (rec.duplicate_group_id, rec.topic)
        if collapse_duplicates and group in seen_groups:
            continue
        seen_records.add(chunk.record_id)
        seen_groups.add(group)
        out.append(RecordCandidate(chunk.record_id, sc.chunk_idx, sc.score))
        if len(out) >= limit:
            break
    return out


def validate_top_k(top_k: int) -> None:
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")


def to_hits(
    store: CorpusStore, cands: Sequence[RecordCandidate], retriever: str, index_version: str | None
) -> list[RetrievalHit]:
    return [store.make_hit(c, i, retriever, index_version) for i, c in enumerate(cands, start=1)]
