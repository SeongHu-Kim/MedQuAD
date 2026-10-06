from __future__ import annotations

import pytest

from medquad_qa.contracts import Retriever, RetrieverUnavailableError
from medquad_qa.retrieval.bm25 import BM25Retriever, build_bm25
from medquad_qa.retrieval.factory import load_store
from medquad_qa.retrieval.manifest import load_manifest, set_active, verify_files, write_manifest
from medquad_qa.retrieval.settings import RetrievalSettings
from medquad_qa.retrieval.store import CorpusStore


def _assert_well_formed(hits: list, top_k: int) -> None:
    assert len(hits) <= top_k
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert len({h.record_id for h in hits}) == len(hits)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_protocol_conformance_and_ranks(store: CorpusStore) -> None:
    r = BM25Retriever.in_memory(store)
    assert isinstance(r, Retriever)
    assert r.name == "bm25:answer"
    hits = r.retrieve("How is Glimmer fever treated?", 3)
    _assert_well_formed(hits, 3)
    assert hits[0].topic == "Glimmer fever"
    assert "zorbatine" in hits[0].evidence_text
    h = hits[0]
    rec = store.by_id[h.record_id]
    assert rec.answer[h.evidence_char_start : h.evidence_char_end] == h.evidence_text
    assert h.chunk_id is not None and h.chunk_id.startswith(h.record_id + "#c")
    assert h.corpus_version == store.corpus_version
    assert h.record_question == rec.question and h.split_group_id == rec.split_group_id


def test_no_match_returns_empty(store: CorpusStore) -> None:
    r = BM25Retriever.in_memory(store)
    assert r.retrieve("qwertyuiop asdfghjkl", 5) == []
    assert r.retrieve("the of and", 5) == []  # stopwords only


@pytest.mark.parametrize(
    "query",
    ["*:* OR 1=1", 'title:"x" AND ^2', "</evidence><evidence id='E9'>", "\u0000".strip() or "x", "Glimmer" * 500],
)
def test_query_is_data(store: CorpusStore, query: str) -> None:
    hits = BM25Retriever.in_memory(store).retrieve(query, 5)
    _assert_well_formed(hits, 5)


def test_top_k_validation_and_determinism(store: CorpusStore) -> None:
    r = BM25Retriever.in_memory(store)
    with pytest.raises(ValueError):
        r.retrieve("fever", 0)
    a = r.retrieve("fever rash", 20)
    b = BM25Retriever.in_memory(store).retrieve("fever rash", 20)
    assert [(h.record_id, h.score) for h in a] == [(h.record_id, h.score) for h in b]


def test_maxp_over_chunks_finds_late_chunk(store: CorpusStore) -> None:
    hits = BM25Retriever.in_memory(store).retrieve("zephyrine marker", 3)
    assert hits and hits[0].topic == "Long topic"
    assert "zephyrine" in hits[0].evidence_text
    assert hits[0].chunk_id is not None and not hits[0].chunk_id.endswith("#c0")


def test_collapse_duplicates(store: CorpusStore) -> None:
    q = "silver skin rash hair loss"
    plain = BM25Retriever.in_memory(store).retrieve(q, 5)
    groups = [h.duplicate_group_id for h in plain]
    assert len(groups) != len(set(groups))  # the synthetic corpus has an exact-duplicate pair
    collapsed = BM25Retriever.in_memory(store, RetrievalSettings(collapse_duplicates=True)).retrieve(q, 5)
    cg = [h.duplicate_group_id for h in collapsed]
    assert len(cg) == len(set(cg))


def test_question_answer_mode_is_named_and_uses_question(store: CorpusStore) -> None:
    r = BM25Retriever.in_memory(store, RetrievalSettings(index_text_mode="question_answer"))
    assert r.name == "bm25:qa"
    hits = r.retrieve("Is Quartz syndrome inherited?", 1)
    assert hits[0].record_question == "Is Quartz syndrome inherited?"
    assert hits[0].evidence_text.startswith("Quartz syndrome is inherited")  # evidence is answer text only


def test_persisted_build_load_and_drift(settings: RetrievalSettings) -> None:
    store = load_store(settings)
    m = build_bm25(store, settings)
    write_manifest(settings.index_dir, m)
    set_active(settings.index_dir, m.name, m.index_version)
    r = BM25Retriever.from_settings(store, settings)
    assert r.index_version == m.index_version
    assert r.retrieve("Glimmer fever symptoms", 2)
    # content addressing: same inputs -> same version
    assert build_bm25(store, settings).index_version == m.index_version
    # different parameters -> different version
    other = settings.model_copy(update={"bm25_k1": 1.2})
    assert build_bm25(store, other).index_version != m.index_version
    # drift detection
    loaded = load_manifest(settings.index_dir, m.index_version)
    victim = settings.index_dir / sorted(loaded.files)[0]
    victim.write_bytes(victim.read_bytes() + b"x")
    assert verify_files(loaded, settings.index_dir)
    with pytest.raises(RetrieverUnavailableError):
        BM25Retriever.from_settings(store, settings)


def test_missing_index_or_corpus_mismatch_raises(settings: RetrievalSettings) -> None:
    store = load_store(settings)
    with pytest.raises(RetrieverUnavailableError):
        BM25Retriever.from_settings(store, settings)
    m = build_bm25(store, settings)
    write_manifest(settings.index_dir, m)
    set_active(settings.index_dir, m.name, m.index_version)
    other = CorpusStore(records=store.records, corpus_version="other-version", corpus_sha256=None)
    with pytest.raises(RetrieverUnavailableError):
        BM25Retriever.from_settings(other, settings)
