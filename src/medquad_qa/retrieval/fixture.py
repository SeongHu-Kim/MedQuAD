"""Fixture retriever: serves pre-specified evidence (for conflict/injection evaluation fixtures and tests)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from medquad_qa.contracts import RetrievalHit


class FixtureRetriever:
    """Returns the given hits (re-ranked 1..n, truncated to top_k), regardless of the query.

    ``hits`` may be a fixed list or a function ``query -> list[RetrievalHit]``.
    """

    def __init__(
        self,
        hits: Sequence[RetrievalHit] | Callable[[str], Sequence[RetrievalHit]],
        *,
        name: str = "fixture",
        corpus_version: str = "fixture",
        index_version: str | None = None,
    ) -> None:
        self._hits = hits
        self.name = name
        self.corpus_version = corpus_version
        self.index_version = index_version

    def retrieve(self, query: str, top_k: int) -> list[RetrievalHit]:
        hits = list(self._hits(query) if callable(self._hits) else self._hits)[:top_k]
        return [h.model_copy(update={"rank": i}) for i, h in enumerate(hits, start=1)]


def make_hit(
    record_id: str,
    evidence_text: str,
    *,
    rank: int = 1,
    score: float = 1.0,
    retriever: str = "fixture",
    corpus_version: str = "fixture",
    **extra: object,
) -> RetrievalHit:
    """Convenience constructor for synthetic hits in tests and fixtures."""
    return RetrievalHit(
        record_id=record_id,
        rank=rank,
        score=score,
        retriever=retriever,
        evidence_text=evidence_text,
        corpus_version=corpus_version,
        **extra,
    )
