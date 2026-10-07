"""Mixed (closed-book + RAG-cited) SFT builder on synthetic records with an offline hashing embedder."""

from __future__ import annotations

import re
from typing import Any

import pytest
from test_training_answerability import _records

from medquad_qa.contracts.qa import RetrievalHit
from medquad_qa.contracts.records import MedicalRecord
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


def _assert_pipeline_parity(
    tok: Any, by_id: dict[str, MedicalRecord], examples: list[Any], rows: list[dict[str, Any]], k: int
) -> int:
    """For every RAG row, RagPipeline.answer() must send the generator exactly the training prompt tokens."""
    from medquad_qa.contracts.qa import QARequest
    from medquad_qa.models.fake import FakeGenerator
    from medquad_qa.rag.budget import EVIDENCE_TRUNCATED
    from medquad_qa.rag.gate import Gate, HeuristicGate
    from medquad_qa.rag.pipeline import RagPipeline
    from medquad_qa.retrieval.corpus import chunk_record

    checked = 0
    for ex, row in zip(examples, rows, strict=True):
        if row["format"] == "closed_book":
            continue
        q = by_id[row["record_id"]]
        chunk_ids = row.get("evidence_chunk_ids") or [None] * len(row["evidence_record_ids"])
        hits = []
        for rank, (rid, cid) in enumerate(zip(row["evidence_record_ids"], chunk_ids, strict=True), start=1):
            rec = by_id[rid]
            chunks = chunk_record(rec)
            if cid is None:
                assert len(chunks) == 1  # synthetic answers are one chunk, so the block text is unambiguous
                chunk = chunks[0]
            else:
                chunk = next(c for c in chunks if c.chunk_id == cid)
            hits.append(
                RetrievalHit(
                    record_id=rid,
                    rank=rank,
                    score=1.0 / rank,
                    retriever="fixture:train-parity",
                    evidence_text=chunk.text,
                    chunk_id=chunk.chunk_id,
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
        response = pipeline.answer(QARequest(question=q.question, experiment_mode="rag", top_k=k), "parity-1")

        assert not response.abstained and EVIDENCE_TRUNCATED not in response.warnings
        assert response.retrieved_record_ids == row["evidence_record_ids"]
        assert len(generator.calls) == 1
        served = [m.model_dump() for m in generator.calls[0]]
        served_ids = tok.apply_chat_template(served, add_generation_prompt=True, tokenize=True, return_dict=False)
        served_ids = list(served_ids["input_ids"] if hasattr(served_ids, "keys") else served_ids)
        prompt_len = sum(1 for t in ex.labels if t == IGNORE_INDEX)
        assert served_ids == ex.input_ids[:prompt_len]  # identical prompt tokens at train and serve time
        checked += 1
    return checked


def test_train_serve_rag_prompt_parity_through_pipeline(tiny_tokenizer: Any) -> None:
    """The v2 training prompt must equal, token for token, what RagPipeline.answer() sends to the generator
    for the same question and evidence (serving path: safety -> retrieve -> gate off -> _step_prompt_rag)."""
    plan = MixPlan(n_rag_answerable=3, n_rag_insufficient=0, n_closed_book=0, k=3, max_target_tokens=400)
    recs, examples, _, rows = _build(tiny_tokenizer, plan)
    assert rows and all(r["format"] == "rag_answerable" for r in rows)
    assert _assert_pipeline_parity(tiny_tokenizer, {r.record_id: r for r in recs}, examples, rows, plan.k) == 3


# ---------------------------------------------------------------------------------------------- sft-mix-v2 pin
V2_GOLDEN_SHA256 = "89ba82c48335f2a4e2c523b8446c172559a6bbf7235c81e0b1c7826b44954074"  # pre-v2b code, same fixture


def test_v2_output_unchanged_when_v2b_keys_absent(tiny_tokenizer: Any) -> None:
    import hashlib
    import json

    _, examples, report, rows = _build(tiny_tokenizer)
    h = hashlib.sha256()
    for ex, row in zip(examples, rows, strict=True):
        h.update(json.dumps([row, ex.input_ids, ex.labels], sort_keys=True).encode())
    assert h.hexdigest() == V2_GOLDEN_SHA256
    d = report.to_dict()
    assert "v2b" not in d and d["mix_version"] == "sft-mix-v2"
    assert set(d["plan"]) == {
        "n_rag_answerable",
        "n_rag_insufficient",
        "n_closed_book",
        "k",
        "max_target_tokens",
        "candidates",
        "seed",
    }


# ---------------------------------------------------------------------------------------------- sft-mix-v2b
V2B_QUESTIONS = {
    "information": "What is (are) {t} ?",
    "symptoms": "What are the symptoms of {t} ?",
    "treatment": "What are the treatments for {t} ?",
    "causes": "What causes {t} ?",
    "inheritance": "Is {t} inherited ?",
    "frequency": "How many people are affected by {t} ?",
}
V2B_ANSWERS = {  # distinct vocabulary per aspect so the overlap guards behave predictably
    "information": "is a rare condition described in medical references with varied presentation",
    "symptoms": "signs include fever cough fatigue rash headache nausea dizziness",
    "treatment": "managed with rest fluids medicine therapy surgery physiotherapy counselling",
    "causes": "results from mutation variant pathway enzyme deficiency protein",
    "inheritance": "passed through families dominant recessive pattern parent offspring",
    "frequency": "occurs rarely estimated prevalence worldwide population births",
}
V2B_LEADS = {  # each answer repeats its question's content words, as real MedQuAD answers usually do
    "information": "overview of",
    "symptoms": "The symptoms of",
    "treatment": "The treatments for",
    "causes": "The causes of",
    "inheritance": "How it is inherited in",
    "frequency": "How many are affected by",
}
V2B_PLAN_KW: dict[str, Any] = {
    "same_topic_weights": [23.5, 21.5, 17.3, 22.2, 15.5],
    "sentinel_disallowed_qtypes": ["information", "other", "support_groups"],
    "sentinel_excluded_qtypes": {
        "causes": ["inheritance", "information"],
        "inheritance": ["causes", "information"],
        "symptoms": ["information"],
        "treatment": ["information"],
        "frequency": ["information"],
    },
    "mix_version": "sft-mix-v2b",
}


def _v2b_records(n_topics: int = 14, start: int = 5000) -> list[MedicalRecord]:
    from training_fixtures import synthetic_record

    out, i = [], start
    for k in range(n_topics):
        topic = f"cond{k}x"
        qtypes = list(V2B_QUESTIONS) if k % 7 else ["symptoms"]  # every 7th topic has one qtype: zero candidates
        for qt in qtypes:
            question = V2B_QUESTIONS[qt].format(t=topic)
            answer = f"{V2B_LEADS[qt]} {topic}: {V2B_ANSWERS[qt]}."
            rec = synthetic_record(i, topic, "symptoms", answer, group=f"g-v2b-{k}")
            out.append(rec.model_copy(update={"question": question, "question_raw": question, "question_type": qt}))
            i += 1
    return out


def _v2b_plan(**overrides: Any) -> MixPlan:
    kw = {"n_rag_answerable": 18, "n_rag_insufficient": 10, "n_sentinel_info": 2, "n_closed_book": 8, "k": 4}
    kw = {**V2B_PLAN_KW, **kw, **overrides}  # overrides win (e.g. mix_version for v2c)
    return MixPlan(max_target_tokens=400, **kw)


def _build_v2b(tok: Any, plan: MixPlan | None = None, max_seq_len: int = 4096) -> tuple[Any, Any, Any, Any]:
    recs = _v2b_records()
    examples, report, rows = MixedSFTBuilder(tok, HashingEmbedder(), max_seq_len=max_seq_len).build(
        recs, plan or _v2b_plan()
    )
    return recs, examples, report, rows


def test_v2b_manifest_fields_consistent(tiny_tokenizer: Any) -> None:
    recs, examples, report, rows = _build_v2b(tiny_tokenizer)
    plan = _v2b_plan()
    assert len(rows) == len(examples) and len({r["record_id"] for r in rows}) == len(rows)
    assert report.to_dict()["mix_version"] == "sft-mix-v2b" and "v2b" in report.to_dict()
    for r in rows:
        if r["format"] == "closed_book":
            assert r["evidence_record_ids"] == [] and r["evidence_chunk_ids"] == []
            continue
        assert len(r["evidence_record_ids"]) == len(r["evidence_chunk_ids"]) == len(r["evidence_on_topic"])
        assert r["on_topic_total"] == sum(r["evidence_on_topic"])
        cap = plan.k - 1 if r["format"] == "rag_answerable" else plan.k
        assert r["same_topic_realised_final"] <= r["same_topic_realised_after_guards"]
        assert (
            r["same_topic_realised_after_guards"]
            == min(r["same_topic_drawn"], r["same_topic_available_after_guards"], cap)
            or r["sentinel_kind"] == "sentinel_info_offtopic"
        )
        if r["format"] == "rag_answerable":
            assert r["gold_slot"] == r["evidence_record_ids"].index(r["record_id"]) + 1
            assert r["on_topic_total"] == 1 + r["same_topic_realised_final"]
            assert 0 <= r["same_topic_drawn"] <= 4
        else:
            assert r["gold_slot"] is None and r["record_id"] not in r["evidence_record_ids"]
            assert r["on_topic_total"] == r["same_topic_realised_final"]


def test_v2b_evidence_rules_and_citations(tiny_tokenizer: Any) -> None:
    """Rewrite of the evidence-pool test for v2b: each non-gold block is either off-group (and not on-topic)
    or same group + same topic_key with a different question_type, different duplicate group, text != gold."""
    from medquad_qa.data.normalize import topic_key

    recs, examples, _, rows = _build_v2b(tiny_tokenizer)
    by_id = {r.record_id: r for r in recs}
    for ex, row in zip(examples, rows, strict=True):
        if row["format"] == "closed_book":
            continue
        q = by_id[row["record_id"]]
        for rid, on_topic in zip(row["evidence_record_ids"], row["evidence_on_topic"], strict=True):
            if rid == q.record_id:
                continue
            b = by_id[rid]
            same = b.split_group_id == q.split_group_id and topic_key(b.topic) == topic_key(q.topic)
            assert on_topic == same
            if same:
                assert b.question_type != q.question_type and b.duplicate_group_id != q.duplicate_group_id
                assert " ".join(b.answer.split()) != " ".join(q.answer.split())
            else:
                assert b.split_group_id != q.split_group_id
        hits = [
            RetrievalHit(record_id=e, rank=i + 1, score=0.0, retriever="t", evidence_text="x", corpus_version="t")
            for i, e in enumerate(row["evidence_record_ids"])
        ]
        check = validate_citations(_target_text(tiny_tokenizer, ex), hits)
        if row["format"] == "rag_answerable":
            assert check.cited_record_ids == [q.record_id] and not check.invalid_ids
        else:
            assert check.has_sentinel and not check.cited_record_ids


def test_v2b_cue_broken_in_both_classes(tiny_tokenizer: Any) -> None:
    _, _, report, rows = _build_v2b(tiny_tokenizer)
    ans = [r for r in rows if r["format"] == "rag_answerable"]
    sen = [r for r in rows if r["sentinel_kind"] == "sentinel_same_topic"]
    assert any(r["on_topic_total"] >= 2 for r in ans)
    assert sen and all(r["on_topic_total"] >= 1 for r in sen)  # same-topic sentinels always show on-topic blocks
    cue = report.v2b["cue_check"]["excluding_zero_candidate"]
    assert cue["rule_accuracy"] < 1.0  # on-topic count alone no longer separates answer from refuse


def test_v2b_sentinel_qtype_rules(tiny_tokenizer: Any) -> None:
    recs, _, report, rows = _build_v2b(tiny_tokenizer)
    by_id = {r.record_id: r for r in recs}
    excluded = V2B_PLAN_KW["sentinel_excluded_qtypes"]
    for r in rows:
        if r["format"] != "rag_insufficient":
            continue
        if r["sentinel_kind"] == "sentinel_info_offtopic":
            assert r["question_type"] == "information" and r["on_topic_total"] == 0
            continue
        assert r["question_type"] not in V2B_PLAN_KW["sentinel_disallowed_qtypes"]
        assert r["same_topic_available_after_guards"] >= 1
        for rid, on_topic in zip(r["evidence_record_ids"], r["evidence_on_topic"], strict=True):
            if on_topic:
                assert by_id[rid].question_type not in excluded.get(r["question_type"], [])
    counts = report.to_dict()["counts"]
    assert counts["rag_insufficient"] == 10
    assert sum(r["sentinel_kind"] == "sentinel_info_offtopic" for r in rows) == 2
    assert report.v2b["g7_drops_by_pair"]  # the pair table actually removed candidates
    assert report.v2b["assignment_skips"].get("sentinel_disallowed_qtype", 0) > 0


def test_v2b_guards_g5_g6_drop_overlapping_candidates() -> None:
    from training_fixtures import synthetic_record

    from medquad_qa.training.sft_rag_data import _Ctx

    def rec(i: int, qt: str, answer: str, dup: str) -> MedicalRecord:
        q = V2B_QUESTIONS[qt].format(t="condz")
        r = synthetic_record(i, "condz", "symptoms", answer, group="g-z")
        return r.model_copy(update={"question": q, "question_type": qt, "duplicate_group_id": dup})

    # gold mentions both question words, so a block covering them equally well (ties drop) is the G6 case
    gold = rec(1, "symptoms", "The symptoms of condz include fever cough fatigue rash headache nausea.", "d1")
    near_dup = rec(2, "treatment", "The symptoms of condz include fever cough fatigue rash headache pallor.", "d2")
    covers = rec(3, "frequency", "symptoms of condz symptoms condz symptoms condz.", "d3")
    clean = rec(4, "treatment", "condz managed with rest fluids medicine therapy surgery physiotherapy.", "d4")
    same_dup = rec(5, "causes", "condz results from mutation variant pathway enzyme deficiency.", "d1")
    plan = MixPlan(n_rag_answerable=0, n_rag_insufficient=0, n_closed_book=0, **V2B_PLAN_KW)
    ctx = _Ctx.make([gold, near_dup, covers, clean, same_dup], plan)
    cands, drops, _ = ctx.candidates(gold, sentinel=True)  # G6 applies to sentinel questions only
    assert set(cands) == {clean.record_id}
    assert drops["G5_gold_jaccard"] == 1 and drops["G6_coverage_ge_gold"] == 1 and drops["G2_gold_dup_group"] == 1


def test_v2b_g6_exempts_answerable_but_drops_for_sentinels() -> None:
    from training_fixtures import synthetic_record

    from medquad_qa.training.sft_rag_data import _Ctx

    def rec(i: int, qt: str, answer: str) -> MedicalRecord:
        q = V2B_QUESTIONS[qt].format(t="condw")
        r = synthetic_record(i, "condw", "symptoms", answer, group="g-w")
        return r.model_copy(update={"question": q, "question_type": qt, "duplicate_group_id": f"dw{i}"})

    gold = rec(1, "symptoms", "The symptoms of condw include fever cough fatigue rash headache nausea.")
    # passes G1-G5 (different qtype and duplicate group, low Jaccard) but covers the question as well as gold
    covers = rec(2, "frequency", "symptoms condw occur rarely estimated prevalence worldwide population.")
    plan = MixPlan(n_rag_answerable=0, n_rag_insufficient=0, n_closed_book=0, **V2B_PLAN_KW)
    ctx = _Ctx.make([gold, covers], plan)
    assert ctx._coverage(gold.question, covers.answer) >= ctx._coverage(gold.question, gold.answer)
    kept, drops_a, _ = ctx.candidates(gold, sentinel=False)
    assert set(kept) == {covers.record_id} and drops_a["G6_coverage_ge_gold"] == 0
    dropped, drops_s, _ = ctx.candidates(gold, sentinel=True)
    assert dropped == {} and drops_s["G6_coverage_ge_gold"] == 1


def test_v2b_answerable_g6_count_is_zero(tiny_tokenizer: Any) -> None:
    _, _, report, _ = _build_v2b(tiny_tokenizer)
    assert report.v2b["guard_drops_candidate_chunks"]["answerable"]["G6_coverage_ge_gold"] == 0


def test_v2b_deterministic_and_selection_modes(tiny_tokenizer: Any) -> None:
    rows_a = _build_v2b(tiny_tokenizer)[3]
    assert _build_v2b(tiny_tokenizer)[3] == rows_a
    plan_r = _v2b_plan(sentinel_selection="random")
    rows_r = _build_v2b(tiny_tokenizer, plan_r)[3]
    assert _build_v2b(tiny_tokenizer, plan_r)[3] == rows_r
    with pytest.raises(ValueError, match="sentinel_selection"):
        _build_v2b(tiny_tokenizer, _v2b_plan(sentinel_selection="best"))


def test_v2b_zero_candidate_answerable_falls_back_to_offgroup(tiny_tokenizer: Any) -> None:
    _, _, report, rows = _build_v2b(
        tiny_tokenizer, _v2b_plan(n_rag_answerable=60, n_rag_insufficient=0, n_sentinel_info=0, n_closed_book=0)
    )
    zero = [r for r in rows if r["format"] == "rag_answerable" and r["same_topic_available_after_guards"] == 0]
    assert zero and all(
        r["on_topic_total"] == 1 and r["n_offgroup_final"] == len(r["evidence_record_ids"]) - 1 for r in zero
    )
    assert report.v2b["zero_candidate"]["rag_answerable"] == len(zero)


def test_v2b_length_drops_remove_offgroup_first(tiny_tokenizer: Any) -> None:
    plan = _v2b_plan(n_rag_answerable=20, n_rag_insufficient=0, n_sentinel_info=0, n_closed_book=0)
    lengths = sorted(len(e.input_ids) for e in _build_v2b(tiny_tokenizer, plan)[1])
    _, examples, _, rows = _build_v2b(tiny_tokenizer, plan, max_seq_len=lengths[len(lengths) // 2] - 1)
    dropped_rows = [r for r in rows if r["dropped_for_length"]]
    assert dropped_rows
    for r in rows:
        assert r["record_id"] in r["evidence_record_ids"]  # gold never dropped
        if r["dropped_for_length"].get("same_topic"):
            assert r["n_offgroup_final"] == 0  # same-topic blocks are only dropped once off-group ones are gone
        assert r["n_offgroup_final"] == r["n_offgroup_initial"] - r["dropped_for_length"].get("offgroup", 0)
        assert r["same_topic_realised_final"] == r["same_topic_realised_after_guards"] - r["dropped_for_length"].get(
            "same_topic", 0
        )


def test_v2b_train_serve_prompt_parity(tiny_tokenizer: Any) -> None:
    recs, examples, _, rows = _build_v2b(tiny_tokenizer)
    n = _assert_pipeline_parity(tiny_tokenizer, {r.record_id: r for r in recs}, examples, rows, 4)
    assert n == sum(r["format"] != "closed_book" for r in rows) > 0


def test_v2b_report_plan_vs_realised_and_exact_histograms(tiny_tokenizer: Any) -> None:
    _, _, report, rows = _build_v2b(tiny_tokenizer)
    v = report.v2b
    pvr = v["plan_vs_realised"]
    assert pvr["answerable"]["planned"] == 18 and pvr["sentinel_same_topic"]["planned"] == 8
    assert pvr["sentinel_info_offtopic"]["planned"] == 2 and pvr["closed_book"]["planned"] == 8
    for k, d in pvr.items():
        assert d["realised"] <= d["assigned"] <= d["planned"], k
    assert pvr["answerable"]["realised"] == report.to_dict()["counts"]["rag_answerable"]
    hist = v["on_topic_total_hist"]
    assert all(set(h) == {"0", "1", "2", "3", "4", "5"} for h in hist.values())
    assert sum(hist["rag_answerable"].values()) == pvr["answerable"]["realised"]
    zero_same = sum(r["sentinel_kind"] == "sentinel_same_topic" and r["on_topic_total"] == 0 for r in rows)
    assert v["sentinel_same_topic_zero_on_topic_after_length_drops"] == zero_same
    for r in rows:
        if r["format"] != "closed_book":
            assert "n_offgroup" not in r and r["n_offgroup_final"] <= r["n_offgroup_initial"]


# ---------------------------------------------------------------------------------------------- v2b pin, v2c
V2B_GOLDEN_ROWS_SHA256 = "b54243c7e9af27cec272335e8f220c661d7360d02898a283c1c97ffff2877a3e"  # pre-v2c code
V2B_GOLDEN_REPORT_SHA256 = "057b401ffa8567f1238cec736caa522526b19f1f021cc09f722b6a56764d6e35"


def test_v2b_output_unchanged_by_v2c_switch(tiny_tokenizer: Any) -> None:
    import hashlib
    import json

    _, examples, report, rows = _build_v2b(tiny_tokenizer)
    h = hashlib.sha256()
    for ex, row in zip(examples, rows, strict=True):
        h.update(json.dumps([row, ex.input_ids, ex.labels], sort_keys=True).encode())
    assert h.hexdigest() == V2B_GOLDEN_ROWS_SHA256
    report_json = json.dumps(report.to_dict(), sort_keys=True, default=str).encode()
    assert hashlib.sha256(report_json).hexdigest() == V2B_GOLDEN_REPORT_SHA256


def test_v2c_sentinels_offtopic_only(tiny_tokenizer: Any) -> None:
    plan = _v2b_plan(sentinel_same_topic=False, mix_version="sft-mix-v2c")
    recs, examples, report, rows = _build_v2b(tiny_tokenizer, plan)
    by_id = {r.record_id: r for r in recs}
    sen = [r for r in rows if r["format"] == "rag_insufficient"]
    ans = [r for r in rows if r["format"] == "rag_answerable"]
    assert sen and all(r["on_topic_total"] == 0 and r["same_topic_realised_final"] == 0 for r in sen)
    assert {r["sentinel_kind"] for r in sen} == {"sentinel_offtopic", "sentinel_info_offtopic"}
    for r in sen:
        if r["sentinel_kind"] == "sentinel_offtopic":
            assert r["question_type"] not in V2B_PLAN_KW["sentinel_disallowed_qtypes"]
        else:
            assert r["question_type"] == "information"
        assert all(by_id[e].split_group_id != by_id[r["record_id"]].split_group_id for e in r["evidence_record_ids"])
    assert any(r["same_topic_realised_final"] > 0 for r in ans)  # answerable prompts keep v2b same-topic blocks
    v = report.v2b
    assert "sentinel_no_candidates_after_guards" not in v["assignment_skips"]
    pvr = v["plan_vs_realised"]
    assert "sentinel_same_topic" not in pvr
    assert pvr["sentinel_offtopic"]["planned"] == pvr["sentinel_offtopic"]["realised"] == 8
    assert pvr["sentinel_info_offtopic"]["planned"] == pvr["sentinel_info_offtopic"]["realised"] == 2
    assert v["guard_drops_candidate_chunks"]["sentinel_offtopic"] == {
        g: 0 for g in v["guard_drops_candidate_chunks"]["sentinel_offtopic"]
    }
    assert v["on_topic_total_hist"]["rag_insufficient"]["0"] == len(sen)
    assert report.to_dict()["mix_version"] == "sft-mix-v2c" and report.plan["sentinel_same_topic"] is False
    assert _assert_pipeline_parity(tiny_tokenizer, by_id, examples, rows, 4) == len(sen) + len(ans)


def test_v2c_config_matches_v2b_except_sentinel_switch() -> None:
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[2] / "configs" / "training"
    b = yaml.safe_load((root / "mix_v2b.yaml").read_text())
    c = yaml.safe_load((root / "mix_v2c.yaml").read_text())
    assert c["mix_version"] == "sft-mix-v2c" and c["v2b"].pop("sentinel_same_topic") is False
    b.pop("mix_version"), c.pop("mix_version")
    assert b == c
