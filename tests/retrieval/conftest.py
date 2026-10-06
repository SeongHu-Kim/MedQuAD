"""Fixtures for retrieval tests. All data is SYNTHETIC (medquad_qa.retrieval.testing)."""

from __future__ import annotations

from pathlib import Path

import pytest

from medquad_qa.retrieval.settings import RetrievalSettings
from medquad_qa.retrieval.store import CorpusStore
from medquad_qa.retrieval.testing import SYNTHETIC_CORPUS_VERSION, synthetic_records, write_synthetic_corpus


@pytest.fixture
def store() -> CorpusStore:
    return CorpusStore(records=synthetic_records(), corpus_version=SYNTHETIC_CORPUS_VERSION, corpus_sha256=None)


@pytest.fixture
def corpus_files(tmp_path: Path) -> tuple[Path, Path]:
    return write_synthetic_corpus(tmp_path / "data")


@pytest.fixture
def settings(tmp_path: Path, corpus_files: tuple[Path, Path]) -> RetrievalSettings:
    corpus, manifest = corpus_files
    return RetrievalSettings(
        retriever="bm25",
        corpus_path=corpus,
        corpus_manifest_path=manifest,
        index_dir=tmp_path / "indexes",
        qdrant_path=tmp_path / "qdrant",
    )
