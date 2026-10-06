"""Read-side helpers for consumers (retrieval, models, evaluation): load and integrity-check outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medquad_qa.contracts import ContractViolationError, MedicalRecord
from medquad_qa.data.config import REPO_ROOT
from medquad_qa.data.ingest import file_sha256

CORPUS_MANIFEST = "data/manifests/corpus_manifest.json"
EXPORTS_MANIFEST = "data/manifests/exports_manifest.json"


@dataclass(frozen=True)
class LoadedRecords:
    records: list[MedicalRecord]
    corpus_version: str
    sha256: str
    split_version: str | None = None


def load_records(path: Path) -> list[MedicalRecord]:
    """Parse a JSONL file of MedicalRecord lines (corpus or a split export), validating each line."""
    with path.open(encoding="utf-8") as fh:
        return [MedicalRecord.model_validate_json(line) for line in fh if line.strip()]


def _manifest(root: Path, rel: str) -> dict[str, Any]:
    path = root / rel
    if not path.exists():
        raise ContractViolationError(f"missing manifest {rel}; run `python -m medquad_qa.data build`")
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _checked(path: Path, expected_sha: str) -> str:
    if not path.exists():
        raise ContractViolationError(f"missing {path}; run `python -m medquad_qa.data build`")
    actual = file_sha256(path)
    if actual != expected_sha:
        raise ContractViolationError(f"{path} sha256 {actual} != manifest {expected_sha}")
    return actual


def load_corpus(root: Path = REPO_ROOT) -> LoadedRecords:
    """Load data/processed/corpus.jsonl after checking its sha256 against corpus_manifest.json."""
    m = _manifest(root, CORPUS_MANIFEST)
    path = root / m["corpus_path"]
    sha = _checked(path, m["corpus_sha256"])
    return LoadedRecords(load_records(path), m["corpus_version"], sha)


def load_split(split: str, root: Path = REPO_ROOT) -> LoadedRecords:
    """Load one split export ('train' | 'validation' | 'test') after checking exports_manifest.json."""
    m = _manifest(root, EXPORTS_MANIFEST)
    if split not in m["splits"]:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(m['splits'])}")
    entry = m["splits"][split]
    path = root / entry["path"]
    sha = _checked(path, entry["sha256"])
    return LoadedRecords(load_records(path), m["corpus_version"], sha, m["split_version"])
