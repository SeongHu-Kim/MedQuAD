"""Corpus loading and chunking (record -> overlapping word-window chunks with char offsets)."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from medquad_qa.contracts import ArtifactUnavailableError, ContractViolationError, MedicalRecord

CHUNKER_VERSION = "chunk-v1"
_WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str  # '<record_id>#c<i>'
    record_id: str
    index: int
    char_start: int  # offsets into MedicalRecord.answer
    char_end: int
    text: str  # == record.answer[char_start:char_end]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_corpus(path: Path) -> list[MedicalRecord]:
    """Load ``corpus.jsonl`` (one MedicalRecord per line). Raises ArtifactUnavailableError if missing."""
    if not path.is_file():
        raise ArtifactUnavailableError(f"corpus file not found: {path}")
    records: list[MedicalRecord] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rec = MedicalRecord.model_validate_json(line)
            except ValueError as exc:
                raise ContractViolationError(f"{path}:{lineno}: invalid MedicalRecord") from exc
            if rec.record_id in seen:
                raise ContractViolationError(f"{path}:{lineno}: duplicate record_id {rec.record_id}")
            seen.add(rec.record_id)
            records.append(rec)
    return records


def read_corpus_version(manifest_path: Path, corpus_path: Path | None = None) -> str:
    """Read ``corpus_version`` from the data-steward manifest; fall back to a content hash of the corpus file."""
    if manifest_path.is_file():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        version = data.get("corpus_version")
        if isinstance(version, str) and version:
            return version
    if corpus_path is not None and corpus_path.is_file():
        return f"unversioned-{sha256_file(corpus_path)[:12]}"
    raise ArtifactUnavailableError(f"corpus manifest not found: {manifest_path}")


def read_corpus_sha256(manifest_path: Path) -> str | None:
    """``corpus_sha256`` recorded by the data-steward manifest, if any."""
    if not manifest_path.is_file():
        return None
    value = json.loads(manifest_path.read_text(encoding="utf-8")).get("corpus_sha256")
    return value if isinstance(value, str) and value else None


def chunk_text(text: str, max_words: int = 200, overlap: int = 40) -> list[tuple[int, int]]:
    """Return (char_start, char_end) windows of <= ``max_words`` words with ``overlap`` words shared.

    Invariants: windows cover every word; consecutive windows overlap by exactly ``overlap`` words
    (except when the text has <= max_words words, giving one window); slices are exact substrings.
    """
    if overlap >= max_words:
        raise ValueError("overlap must be smaller than max_words")
    spans = [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]
    if not spans:
        return []
    stride = max_words - overlap
    windows: list[tuple[int, int]] = []
    start = 0
    while True:
        end = min(start + max_words, len(spans))
        windows.append((spans[start][0], spans[end - 1][1]))
        if end >= len(spans):
            break
        start += stride
    return windows


def chunk_record(record: MedicalRecord, max_words: int = 200, overlap: int = 40) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"{record.record_id}#c{i}",
            record_id=record.record_id,
            index=i,
            char_start=s,
            char_end=e,
            text=record.answer[s:e],
        )
        for i, (s, e) in enumerate(chunk_text(record.answer, max_words, overlap))
    ]


def chunk_corpus(records: Iterable[MedicalRecord], max_words: int = 200, overlap: int = 40) -> list[Chunk]:
    chunks: list[Chunk] = []
    for rec in records:
        chunks.extend(chunk_record(rec, max_words, overlap))
    return chunks


def index_text(chunk: Chunk, record: MedicalRecord, mode: str) -> str:
    """Text that is indexed for a chunk. Evidence returned to callers is always the answer chunk."""
    if mode == "answer":
        return chunk.text
    if mode == "question_answer":
        return f"{record.question}\n{chunk.text}"
    raise ValueError(f"unknown index text mode: {mode}")


def texts_fingerprint(texts: Sequence[str], ids: Sequence[str]) -> str:
    """Order-sensitive sha256 over (id, text) pairs; used for content-addressed index versions."""
    h = hashlib.sha256()
    for cid, text in zip(ids, texts, strict=True):
        h.update(cid.encode())
        h.update(b"\x1f")
        h.update(text.encode())
        h.update(b"\x1e")
    return h.hexdigest()
