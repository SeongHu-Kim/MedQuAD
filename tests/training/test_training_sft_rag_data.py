"""Mixed (closed-book + RAG-cited) SFT builder on synthetic records with an offline hashing embedder."""

from __future__ import annotations

import re
from typing import Any

import pytest
from test_training_answerability import _records

from medquad_qa.contracts.qa import RetrievalHit
from medquad_qa.rag.citations import validate_citations
from medquad_qa.rag.prompts import SENTINEL
from medquad_qa.retrieval.embedding import HashingEmbedder
from medquad_qa.training.sft_data import IGNORE_INDEX
from medquad_qa.training.sft_rag_data import MixedSFTBuilder, MixPlan, cited_target

PLAN = MixPlan(n_rag_answerable=8, n_rag_insufficient=4, n_closed_book=6, k=3, max_target_tokens=400)


def _build(tok: Any, plan: MixPlan = PLAN, max_seq_len: int = 4096) -> tuple[Any, Any, Any, Any]:
    recs = _records("tr", 15, 0)
    builder = MixedSFTBuilder(tok, HashingEmbedder(), max_seq_len=max_seq_len)
    examples, report, rows = builder.build(recs, plan)
    return recs, examples, report, rows


def _target_text(tok: Any, ex: Any) -> str:
    return str(tok.decode([t for t in ex.labels if t != IGNORE_INDEX], skip_special_tokens=True))


def test_cited_target_cites_every_sentence() -> None:
    text, used, total = cited_target("First fact. Second fact!\n- bullet line", "E2", lambda t: True)
    assert text == "First fact. [E2] Second fact! [E2]\n- bullet line [E2]" and used == total == 3
    short, used, total = cited_target("First fact. Second fact.", "E1", lambda t: len(t) < 20)
    assert short == "First fact. [E1]" and (used, total) == (1, 2)


def test_mix_counts_formats_and_disjoint_questions(tiny_tokenizer: Any) -> None:
    recs, examples, report, rows = _build(tiny_tokenizer)
    assert report.counts == {"rag_answerable": 8, "rag_insufficient": 4, "closed_book": 6}
    assert len({r["record_id"] for r in rows}) == len(rows) == len(examples)  # each question used once
    assert {r["record_id"] for r in rows} <= {r.record_id for r in recs if not r.quality_flags}
    assert _build(tiny_tokenizer)[3] == rows  # deterministic


def test_rag_targets_validate_and_evidence_stays_in_pool(tiny_tokenizer: Any) -> None:
    recs, examples, report, rows = _build(tiny_tokenizer)
    by_id = {r.record_id: r for r in recs}
    for ex, row in zip(examples, rows, strict=True):
        q = by_id[row["record_id"]]
        if row["format"] == "closed_book":
            assert row["evidence_record_ids"] == []
            continue
        ev = row["evidence_record_ids"]
        assert all(by_id[e].split_group_id != q.split_group_id for e in ev if e != q.record_id)
        assert all(not by_id[e].quality_flags for e in ev)  # boilerplate never used as evidence
        hits = [
            RetrievalHit(record_id=e, rank=i + 1, score=0.0, retriever="t", evidence_text="x", corpus_version="t")
            for i, e in enumerate(ev)
        ]
        text = _target_text(tiny_tokenizer, ex)
        check = validate_citations(text, hits)
        if row["format"] == "rag_answerable":
            assert q.record_id in ev and len(ev) == PLAN.k
            assert check.cited_record_ids == [q.record_id] and not check.invalid_ids
            assert re.search(r"\[E\d\]", text)
        else:
            assert q.record_id not in ev and text.strip() == SENTINEL and check.has_sentinel
            assert all(by_id[e].topic != q.topic for e in ev)
    assert sum(report.gold_position.values()) == 8


def test_length_budget_drops_distractors_never_gold(tiny_tokenizer: Any) -> None:
    plan = MixPlan(n_rag_answerable=4, n_rag_insufficient=0, n_closed_book=0, k=3, max_target_tokens=60)
    _, roomy, _, _ = _build(tiny_tokenizer, plan, max_seq_len=4096)
    lengths = sorted(len(e.input_ids) for e in roomy)
    _, tight, report, rows = _build(tiny_tokenizer, plan, max_seq_len=lengths[0] - 1)
    assert report.distractors_dropped_for_length > 0 or report.skipped
    for ex, row in zip(tight, rows, strict=True):
        assert len(ex.input_ids) <= lengths[0] - 1
        assert row["record_id"] in row["evidence_record_ids"]  # gold kept


@pytest.mark.parametrize("n", [0])
def test_empty_rag_plan_is_closed_book_only(tiny_tokenizer: Any, n: int) -> None:
    plan = MixPlan(n_rag_answerable=n, n_rag_insufficient=n, n_closed_book=5, k=3)
    _, examples, report, rows = _build(tiny_tokenizer, plan)
    assert report.counts == {"closed_book": 5} and all(r["format"] == "closed_book" for r in rows)


class _FixedRetriever:
    """Returns exactly the training example's evidence, in the same order, as serving RetrievalHits."""

    name = "fixture:train-parity"
    corpus_version = "sft-parity-test"
    index_version = None

    def __init__(self, hits: list[RetrievalHit]) -> None:
        self.hits = hits

    def retrieve(self, query: str, top_k: int) -> list[RetrievalHit]:
        return self.hits[:top_k]


def test_train_serve_rag_prompt_parity_through_pipeline(tiny_tokenizer: Any) -> None:
    """The v2 training prompt must equal, token for token, what RagPipeline.answer() sends to the generator
    for the same question and evidence (serving path: safety -> retrieve -> gate off -> _step_prompt_rag)."""
    from medquad_qa.contracts.qa import QARequest
    from medquad_qa.models.fake import FakeGenerator
    from medquad_qa.rag.budget import EVIDENCE_TRUNCATED
    from medquad_qa.rag.gate import Gate, HeuristicGate
    from medquad_qa.rag.pipeline import RagPipeline
    from medquad_qa.retrieval.corpus import chunk_record

    plan = MixPlan(n_rag_answerable=3, n_rag_insufficient=0, n_closed_book=0, k=3, max_target_tokens=400)
    recs, examples, _, rows = _build(tiny_tokenizer, plan)
    by_id = {r.record_id: r for r in recs}
    assert rows and all(r["format"] == "rag_answerable" for r in rows)

    for ex, row in zip(examples, rows, strict=True):
        q = by_id[row["record_id"]]
        hits = []
        for rank, rid in enumerate(row["evidence_record_ids"], start=1):
            rec = by_id[rid]
            chunks = chunk_record(rec)
            assert len(chunks) == 1  # synthetic answers are one chunk, so the block text is unambiguous
            hits.append(
                RetrievalHit(
                    record_id=rid,
                    rank=rank,
                    score=1.0 / rank,
                    retriever="fixture:train-parity",
                    evidence_text=chunks[0].text,
                    chunk_id=chunks[0].chunk_id,
                    topic=rec.topic,
                    source_name=rec.source_name,
                    corpus_version="sft-parity-test",
                )
            )
        generator = FakeGenerator()  # records the messages it receives; cites [E1] so the pipeline answers
        pipeline = RagPipeline(
            retriever=_FixedRetriever(hits),
            generator_provider=lambda variant, g=generator: g,
            gate=Gate(None, HeuristicGate(), mode="off"),
        )
        response = pipeline.answer(QARequest(question=q.question, experiment_mode="rag", top_k=plan.k), "parity-1")

        assert not response.abstained and EVIDENCE_TRUNCATED not in response.warnings
        assert response.retrieved_record_ids == row["evidence_record_ids"]
        assert len(generator.calls) == 1
        served = [m.model_dump() for m in generator.calls[0]]
        served_ids = tiny_tokenizer.apply_chat_template(
            served, add_generation_prompt=True, tokenize=True, return_dict=False
        )
        served_ids = list(served_ids["input_ids"] if hasattr(served_ids, "keys") else served_ids)
        prompt_len = sum(1 for t in ex.labels if t == IGNORE_INDEX)
        assert served_ids == ex.input_ids[:prompt_len]  # identical prompt tokens at train and serve time
