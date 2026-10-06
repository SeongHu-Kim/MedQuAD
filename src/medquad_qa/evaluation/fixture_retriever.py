"""FixtureRetriever: serves synthetic evidence for conflicting-evidence and injection cases (Track C, E5).

It implements the ``Retriever`` protocol so it can be injected into the RAG pipeline's DI constructor. Fixture
record IDs match the corpus ID pattern (``mq-`` + 16 hex, from a differently-salted sha256) so they pass
citation validation; hits are marked ``source_name='synthetic_fixture'``. Collisions with corpus IDs are
astronomically unlikely and are additionally checked against the corpus when a fixture set is built.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from medquad_qa.contracts.qa import MAX_TOP_K, RetrievalHit

FIXTURE_CORPUS_VERSION = "fixture-evidence-v1"


def fixture_record_id(text: str) -> str:
    return "mq-" + hashlib.sha256(("medquad-fixture-v1\x1f" + text).encode()).hexdigest()[:16]


class FixtureRetriever:
    """Returns pre-registered evidence for exact query strings; [] for anything else."""

    name = "fixture"
    corpus_version = FIXTURE_CORPUS_VERSION
    index_version: str | None = None

    def __init__(self, evidence_by_query: Mapping[str, Sequence[str]]) -> None:
        self._hits: dict[str, list[RetrievalHit]] = {}
        for query, texts in evidence_by_query.items():
            self._hits[query.strip()] = [
                RetrievalHit(
                    record_id=fixture_record_id(t),
                    rank=i,
                    score=1.0 / i,
                    retriever=self.name,
                    evidence_text=t,
                    source_name="synthetic_fixture",
                    quality_flags=["synthetic_fixture"],
                    corpus_version=self.corpus_version,
                )
                for i, t in enumerate(texts, 1)
            ]

    def retrieve(self, query: str, top_k: int) -> list[RetrievalHit]:
        if not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be in 1..{MAX_TOP_K}")
        return list(self._hits.get(query.strip(), []))[:top_k]
