"""Fixtures for RAG tests. All data SYNTHETIC."""

from __future__ import annotations

import pytest

from medquad_qa.retrieval.bm25 import BM25Retriever
from medquad_qa.retrieval.store import CorpusStore
from medquad_qa.retrieval.testing import SYNTHETIC_CORPUS_VERSION, synthetic_records


@pytest.fixture
def store() -> CorpusStore:
    return CorpusStore(records=synthetic_records(), corpus_version=SYNTHETIC_CORPUS_VERSION, corpus_sha256=None)


@pytest.fixture
def retriever(store: CorpusStore) -> BM25Retriever:
    return BM25Retriever.in_memory(store)
