"""Hybrid retrieval: reciprocal-rank fusion (RRF) of BM25 and dense record lists, optional cross-encoder
rerank, and graceful degradation to lexical-only when the dense backend is unavailable."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from medquad_qa.contracts import RetrievalHit, RetrieverUnavailableError
from medquad_qa.retrieval.embedding import resolve_model_snapshot
from medquad_qa.retrieval.settings import MODE_SUFFIX
from medquad_qa.retrieval.store import CorpusStore, RecordCandidate, to_hits, validate_top_k

log = logging.getLogger(__name__)

LEXICAL_FALLBACK = "lexical_fallback"


class RecordSearcher(Protocol):
    name: str
    index_version: str | None

    def search_records(self, query: str, limit: int) -> list[RecordCandidate]: ...


def rrf_fuse(lists: list[list[RecordCandidate]], k: int = 60) -> list[RecordCandidate]:
    """Fuse ranked record lists by RRF: score(d) = sum_i 1 / (k + rank_i(d)).

    The evidence chunk for a fused record comes from the list where it ranks best (earlier list wins
    ties). Output ties are broken by record_id for determinism.
    """
    scores: dict[str, float] = {}
    best: dict[str, tuple[int, int, int]] = {}  # record_id -> (rank, list_idx, chunk_idx)
    for li, lst in enumerate(lists):
        for rank, cand in enumerate(lst, start=1):
            scores[cand.record_id] = scores.get(cand.record_id, 0.0) + 1.0 / (k + rank)
            prev = best.get(cand.record_id)
            if prev is None or (rank, li) < (prev[0], prev[1]):
                best[cand.record_id] = (rank, li, cand.chunk_idx)
    order = sorted(scores, key=lambda rid: (-scores[rid], rid))
    return [RecordCandidate(rid, best[rid][2], scores[rid]) for rid in order]


class CrossEncoderReranker:
    """Optional MiniLM cross-encoder; reorders the top ``depth`` fused candidates by (query, chunk) score."""

    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        *,
        device: str = "auto",
        depth: int = 20,
        allow_download: bool = False,
    ) -> None:
        from sentence_transformers import CrossEncoder

        local, commit = resolve_model_snapshot(model_id, revision, allow_download=allow_download)
        if device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = CrossEncoder(str(local), device=device)
        self.model_id = model_id
        self.revision = commit
        self.depth = depth

    def rerank(self, query: str, cands: list[RecordCandidate], store: CorpusStore) -> list[RecordCandidate]:
        head, tail = cands[: self.depth], cands[self.depth :]
        if not head:
            return cands
        pairs = [(query, store.chunks[c.chunk_idx].text) for c in head]
        scores = [float(s) for s in self._model.predict(pairs, show_progress_bar=False)]
        ranked = sorted(zip(head, scores, strict=True), key=lambda t: (-t[1], t[0].record_id))
        return [RecordCandidate(c.record_id, c.chunk_idx, s) for c, s in ranked] + tail


class HybridRetriever:
    """RRF(bm25, dense). If the dense side raises RetrieverUnavailableError the result is lexical-only,
    named after the BM25 retriever, and ``retrieve_with_info`` reports the ``lexical_fallback`` warning.

    With ``fuse=False`` it is dense-only with the same lexical fallback (``MEDQUAD_RETRIEVER=dense_fallback``);
    results are then named after the dense retriever (``dense:<mode>``)."""

    def __init__(
        self,
        store: CorpusStore,
        lexical: RecordSearcher,
        dense: RecordSearcher | None,
        *,
        mode: str = "answer",
        rrf_k: int = 60,
        candidate_k: int = 50,
        reranker: Any | None = None,
        dense_error: str | None = None,
        fuse: bool = True,
    ) -> None:
        self.store = store
        self.lexical = lexical
        self.dense = dense
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        self.reranker = reranker
        self.dense_error = dense_error
        self.fuse = fuse
        base = "hybrid_rrf" if fuse else "dense"
        if reranker is not None:
            base += "+ce"
        self.name = f"{base}:{MODE_SUFFIX[mode]}"
        self.corpus_version = store.corpus_version
        dense_version = dense.index_version if dense is not None else None
        if not fuse:
            self.index_version = dense_version or lexical.index_version
        else:
            self.index_version = f"{lexical.index_version}+{dense_version}" if dense_version else lexical.index_version

    def retrieve_with_info(self, query: str, top_k: int) -> tuple[list[RetrievalHit], str, list[str]]:
        """Return (hits, retriever_name_actually_used, warnings)."""
        validate_top_k(top_k)
        dense_cands: list[RecordCandidate] | None = None
        if self.dense is not None:
            try:
                dense_cands = self.dense.search_records(query, self.candidate_k)
            except RetrieverUnavailableError as exc:
                log.warning("dense retriever unavailable, using lexical fallback: %s", type(exc).__name__)
        if dense_cands is None:
            lex = self.lexical.search_records(query, max(top_k, self.candidate_k))
            return (
                to_hits(self.store, lex[:top_k], self.lexical.name, self.lexical.index_version),
                self.lexical.name,
                [LEXICAL_FALLBACK],
            )
        if self.fuse:
            fused = rrf_fuse([self.lexical.search_records(query, self.candidate_k), dense_cands], self.rrf_k)
        else:
            fused = dense_cands
        if self.reranker is not None:
            fused = self.reranker.rerank(query, fused, self.store)
        return to_hits(self.store, fused[:top_k], self.name, self.index_version), self.name, []

    def retrieve(self, query: str, top_k: int) -> list[RetrievalHit]:
        return self.retrieve_with_info(query, top_k)[0]
