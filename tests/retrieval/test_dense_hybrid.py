"""Dense + hybrid tests with Qdrant embedded local mode and a deterministic hashing embedder (offline)."""

from __future__ import annotations

from typing import Any

import pytest

from medquad_qa.contracts import Retriever, RetrieverUnavailableError
from medquad_qa.retrieval.bm25 import BM25Retriever, build_bm25
from medquad_qa.retrieval.dense import DenseRetriever, build_dense, open_qdrant, verify_dense
from medquad_qa.retrieval.embedding import HashingEmbedder
from medquad_qa.retrieval.factory import build_retriever, load_store
from medquad_qa.retrieval.hybrid import LEXICAL_FALLBACK, HybridRetriever, rrf_fuse
from medquad_qa.retrieval.manifest import set_active, write_manifest
from medquad_qa.retrieval.settings import RetrievalSettings
from medquad_qa.retrieval.store import CorpusStore, RecordCandidate


@pytest.fixture
def built(settings: RetrievalSettings) -> tuple[RetrievalSettings, CorpusStore, HashingEmbedder, Any]:
    store = load_store(settings)
    emb = HashingEmbedder()
    client = open_qdrant(settings)
    for m in (build_bm25(store, settings), build_dense(store, settings, emb, client)):
        write_manifest(settings.index_dir, m)
        set_active(settings.index_dir, m.name, m.index_version)
    yield settings, store, emb, client
    client.close()


def test_dense_retrieve_and_verify(built: tuple) -> None:
    settings, store, emb, client = built
    r = DenseRetriever.from_settings(store, settings, emb, client)
    assert isinstance(r, Retriever) and r.name == "dense:answer"
    hits = r.retrieve("zorbatine rest fluids", 3)
    assert [h.rank for h in hits] == [1, 2, 3]
    assert "zorbatine" in hits[0].evidence_text
    assert r.retrieve("   ", 3) == []
    from medquad_qa.retrieval.manifest import resolve_active

    assert verify_dense(resolve_active(settings.index_dir, "dense:answer"), client, store) == []


def test_dense_missing_collection_and_embedder_mismatch(built: tuple) -> None:
    settings, store, emb, client = built
    with pytest.raises(RetrieverUnavailableError):
        DenseRetriever.from_settings(store, settings, HashingEmbedder(32), client)
    from medquad_qa.retrieval.manifest import resolve_active

    m = resolve_active(settings.index_dir, "dense:answer")
    client.delete_collection(m.index_version)
    with pytest.raises(RetrieverUnavailableError):
        DenseRetriever.from_settings(store, settings, emb, client)
    assert verify_dense(m, client, store)


def test_qdrant_server_unreachable_raises() -> None:
    s = RetrievalSettings(qdrant_url="http://127.0.0.1:9")  # discard port: nothing listens
    with pytest.raises(RetrieverUnavailableError):
        open_qdrant(s, timeout=1)


def test_rrf_fuse_math_and_tiebreak() -> None:
    a = [RecordCandidate("r1", 0, 9.0), RecordCandidate("r2", 1, 5.0)]
    b = [RecordCandidate("r2", 2, 0.9), RecordCandidate("r3", 3, 0.8)]
    fused = rrf_fuse([a, b], k=60)
    assert [c.record_id for c in fused] == ["r2", "r1", "r3"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[0].chunk_idx == 2  # best rank for r2 is rank 1 in list b
    tie = rrf_fuse([[RecordCandidate("z", 0, 1.0)], [RecordCandidate("a", 1, 1.0)]], k=60)
    assert [c.record_id for c in tie] == ["a", "z"]


def test_hybrid_retrieves_and_names(built: tuple) -> None:
    settings, store, emb, client = built
    lex = BM25Retriever.from_settings(store, settings)
    dense = DenseRetriever.from_settings(store, settings, emb, client)
    h = HybridRetriever(store, lex, dense)
    assert isinstance(h, Retriever) and h.name == "hybrid_rrf:answer"
    hits, name, warnings = h.retrieve_with_info("Glimmer fever treatment zorbatine", 4)
    assert name == "hybrid_rrf:answer" and warnings == []
    assert [x.rank for x in hits] == [1, 2, 3, 4]
    assert hits[0].index_version == f"{lex.index_version}+{dense.index_version}"


class _BrokenDense:
    name = "dense:answer"
    index_version = "dense-x"

    def search_records(self, query: str, limit: int) -> list[RecordCandidate]:
        raise RetrieverUnavailableError("qdrant down")


def test_hybrid_degrades_to_lexical(store: CorpusStore) -> None:
    lex = BM25Retriever.in_memory(store)
    h = HybridRetriever(store, lex, _BrokenDense())
    hits, name, warnings = h.retrieve_with_info("glimmer fever", 3)
    assert name == "bm25:answer" and warnings == [LEXICAL_FALLBACK]
    assert hits and all(x.retriever == "bm25:answer" for x in hits)


class _ReverseReranker:
    def rerank(self, query: str, cands: list[RecordCandidate], store: CorpusStore) -> list[RecordCandidate]:
        return list(reversed(cands))


def test_hybrid_reranker_variant(built: tuple) -> None:
    settings, store, emb, client = built
    lex = BM25Retriever.from_settings(store, settings)
    dense = DenseRetriever.from_settings(store, settings, emb, client)
    plain = HybridRetriever(store, lex, dense).retrieve("glimmer fever", 20)
    rr = HybridRetriever(store, lex, dense, reranker=_ReverseReranker())
    assert rr.name == "hybrid_rrf+ce:answer"
    reranked = rr.retrieve("glimmer fever", 20)
    assert [h.record_id for h in reranked] == [h.record_id for h in reversed(plain)]


def test_factory_hybrid_without_dense_index_degrades(settings: RetrievalSettings) -> None:
    store = load_store(settings)
    m = build_bm25(store, settings)
    write_manifest(settings.index_dir, m)
    set_active(settings.index_dir, m.name, m.index_version)
    bundle = build_retriever(
        settings.model_copy(update={"retriever": "hybrid"}), store=store, embedder=HashingEmbedder()
    )
    assert bundle.retriever is not None
    dense_status = next(s for s in bundle.statuses if s.name == "retriever:dense")
    assert not dense_status.ok and dense_status.degraded and not dense_status.required
    hits, name, warnings = bundle.retriever.retrieve_with_info("glimmer", 2)  # type: ignore[attr-defined]
    assert warnings == [LEXICAL_FALLBACK] and name == "bm25:answer"


def test_factory_full_hybrid(built: tuple) -> None:
    settings, store, emb, client = built
    bundle = build_retriever(
        settings.model_copy(update={"retriever": "hybrid"}), store=store, embedder=emb, qdrant_client=client
    )
    assert bundle.retriever is not None and bundle.retriever.name == "hybrid_rrf:answer"
    assert all(s.ok for s in bundle.statuses)


def test_factory_missing_corpus_never_raises(tmp_path) -> None:
    s = RetrievalSettings(
        retriever="bm25",
        corpus_path=tmp_path / "nope.jsonl",
        corpus_manifest_path=tmp_path / "nope.json",
        index_dir=tmp_path,
    )
    bundle = build_retriever(s)
    assert bundle.retriever is None
    assert bundle.statuses[0].name == "corpus" and not bundle.statuses[0].ok
