"""Offline tests for the E4b comparison module (synthetic responses and examples only)."""

from __future__ import annotations

from typing import Any

import pytest

from medquad_qa.contracts import Citation, EvaluationExample, QAResponse
from medquad_qa.evaluation import e4b_compare as cmp

A, B, G = "mq-" + "a" * 16, "mq-" + "b" * 16, "mq-" + "c" * 16


def ex(eid: str, **kw: Any) -> EvaluationExample:
    base: dict[str, Any] = dict(
        example_id=eid,
        question=f"Synthetic question {eid}?",
        gold_record_ids=[G],
        derived_from_record_ids=[G],
        split_group_id=f"g-{eid}",
        answerable=True,
        label_provenance="synthetic_rule",
        question_provenance="synthetic_rule",
        eval_split="test",
        evaluation_track="corpus_grounded",
        topic="Condition X",
    )
    base.update(kw)
    return EvaluationExample(**base)


def resp(
    eid: str,
    *,
    abstained: bool = False,
    reason: Any = None,
    cites: tuple[str, ...] = (),
    invalid: tuple[str, ...] = (),
    retrieved: tuple[str, ...] = (A, B),
    answer: str = "Synthetic answer text about condition x.",
) -> QAResponse:
    return QAResponse(
        request_id=eid,
        experiment_mode="rag",
        answer=answer,
        abstained=abstained,
        abstention_reason=reason,
        citations=[Citation(record_id=c, evidence_snippet="s") for c in cites],
        retrieved_record_ids=list(retrieved),
        invalid_citation_ids=list(invalid),
        model_version="m",
        prompt_version="p",
        latency_ms=1.0,
    )


def test_provenance_check() -> None:
    p = {k: "x" for k in cmp.PROVENANCE_KEYS}
    m = {"provenance": p, "nli": "n@r", "threshold": 0.5}
    assert cmp.check_provenance(m, dict(m)) == []
    m2 = {"provenance": {**p, "score_e4_py_sha256": "y"}, "nli": "n@r", "threshold": 0.5}
    assert cmp.check_provenance(m, m2) == ["score_e4_py_sha256"]
    assert cmp.check_provenance(m, {**m, "nli": "other"}) == ["nli"]


def test_answer_trained_probe_roles() -> None:
    probe = [
        ex("p1", gold_record_ids=[A]),
        ex("p2", gold_record_ids=[B]),
        ex("p3", gold_record_ids=[G], derived_from_record_ids=[G]),
    ]
    rows = [
        {"record_id": A, "format": "closed_book", "evidence_record_ids": []},
        {"record_id": "mq-" + "d" * 16, "format": "rag_answerable", "evidence_record_ids": [B]},  # B evidence only
        {"record_id": G, "format": "rag_insufficient", "evidence_record_ids": []},  # question only
    ]
    assert cmp.answer_trained_probe_ids(rows, probe) == {"p1": "closed_book"}


def test_copy_rate_and_validity() -> None:
    ev = ["one two three four five six seven eight nine ten"]
    assert cmp.copy_rate(f"one two three four five six seven eight [{A}]", ev) == 1.0
    assert cmp.copy_rate("short answer", ev) is None
    assert cmp.copy_rate("alpha beta gamma delta epsilon zeta eta theta", ev) == 0.0
    assert cmp.citation_validity(resp("x", cites=(A,), invalid=("E9",))) == 0.5
    assert cmp.citation_validity(resp("x")) is None


def test_holm_matches_hand_values() -> None:
    assert cmp.holm({"a": 0.01, "b": 0.04, "c": 0.03}) == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


def test_primary_family_has_exactly_ten_tests_and_detects_difference() -> None:
    a_ex = {f"a{i}": ex(f"a{i}") for i in range(30)}
    c_ex = {
        "c1": ex("c1", notes="general_control:pair=x"),
        "c2": ex(
            "c2",
            answerable=False,
            expected_behavior="abstain",
            expected_abstention_reason="no_relevant_evidence",
            notes="unanswerable:out_of_corpus",
            gold_record_ids=[],
            derived_from_record_ids=[],
        ),
        "c3": ex(
            "c3",
            answerable=False,
            expected_behavior="abstain",
            expected_abstention_reason="no_relevant_evidence",
            notes="unanswerable:fictional",
            gold_record_ids=[],
            derived_from_record_ids=[],
        ),
        "c4": ex(
            "c4",
            answerable=False,
            expected_behavior="abstain",
            expected_abstention_reason="insufficient_evidence",
            notes="hard_negative:missing_qtype=frequency",
            case_type="unanswerable",
        ),
    }
    rag_a = {i: resp(i, cites=(A,)) for i in a_ex}
    v1_a = {i: resp(i, abstained=True, reason="missing_citations") for i in a_ex}
    v2_a = {i: resp(i, cites=(A,)) for i in a_ex}
    rag_c = {i: resp(i, abstained=i != "c1", reason=None if i == "c1" else "insufficient_evidence") for i in c_ex}
    sc = {i: {"ref_coverage": 0.5, "citation_support": 1.0, "unsupported": 0.0} for i in a_ex}
    fam = cmp.primary_family(a_ex, c_ex, rag_a, v1_a, v2_a, rag_c, dict(rag_c), sc, dict(sc))
    assert fam["family_size"] == 10
    assert set(fam["holm_adjusted_p"]) == set(fam["tests"])
    assert fam["tests"]["P2_answered_v2_vs_v1"]["b_only_correct"] == 30  # v2 answers where v1 abstained
    assert fam["holm_adjusted_p"]["P2_answered_v2_vs_v1"] < 0.001
    assert fam["tests"]["P1a_answered"]["p_value"] == 1.0


def test_regression_flag_rule() -> None:
    a_ex = {f"a{i}": ex(f"a{i}") for i in range(40)}
    v1 = {i: {"ref_coverage": 0.6, "ref_unsupported": 0.2} for i in a_ex}
    worse = {i: {"ref_coverage": 0.3, "ref_unsupported": 0.2} for i in a_ex}
    assert cmp.regression_check(a_ex, v1, worse)["flagged"] is True
    assert cmp.regression_check(a_ex, v1, dict(v1))["flagged"] is False


def test_descriptive_topic_analyses() -> None:
    topic_of = {A: "Condition X", B: "Other thing", G: "Condition X"}
    c_ex = {
        "h1": ex(
            "h1",
            answerable=False,
            expected_behavior="abstain",
            expected_abstention_reason="insufficient_evidence",
            notes="hard_negative:missing_qtype=frequency",
            case_type="unanswerable",
            topic="condition-x",
        ),
        "h2": ex(
            "h2",
            answerable=False,
            expected_behavior="abstain",
            expected_abstention_reason="insufficient_evidence",
            notes="hard_negative:missing_qtype=research",
            case_type="unanswerable",
            topic="Condition Z",
        ),
    }
    a_ex = {"a1": ex("a1"), "a2": ex("a2")}
    runs_c = {
        "m": {
            "h1": resp("h1", retrieved=(A,)),
            "h2": resp("h2", abstained=True, reason="insufficient_evidence", retrieved=(B,)),
        }
    }
    runs_a = {"m": {"a1": resp("a1", retrieved=(G, A)), "a2": resp("a2", retrieved=(G, B))}}
    out = cmp.descriptive_topic_analyses(a_ex, c_ex, topic_of, runs_a, runs_c)
    hn = out["i_hard_negative_on_topic"]["m"]
    assert hn["on_topic"]["n"] == 1 and hn["on_topic"]["abstain_recall"] == 0.0  # 'condition-x' folds to topic of A
    assert hn["off_topic"]["n"] == 1 and hn["off_topic"]["abstain_recall"] == 1.0
    ctx = out["ii_track_a_same_topic_context"]["m"]
    assert ctx["same_topic_other"]["n"] == 1 and ctx["no_same_topic_other"]["n"] == 1  # gold G excluded


def test_track_c_descriptive_counts_canary() -> None:
    c_ex = {"x": ex("x", case_type="adversarial", notes="adversarial:canary=CANARYABC")}
    out = cmp.track_c_descriptive(c_ex, {"m": {"x": resp("x", answer="fine CANARYABC")}})
    assert out["m"]["canary_leaked"] == 1 and out["m"]["adversarial_n"] == 1


def test_topic_match_uses_topic_key_and_missing_never_matches() -> None:
    hn = dict(
        answerable=False,
        expected_behavior="abstain",
        expected_abstention_reason="insufficient_evidence",
        notes="hard_negative:missing_qtype=frequency",
        case_type="unanswerable",
    )
    topic_of = {A: "Sjögren's Syndrome", B: "", G: "Other"}
    c_ex = {
        "fmt": ex("fmt", topic="sjogren s  SYNDROME", **hn),  # punctuation/case/diacritics differ, same key
        "empty": ex("empty", topic="", **hn),  # empty item topic vs empty record topic: no match
        "none": ex("none", topic=None, **hn),  # missing item topic vs record missing from topic_of: no match
    }
    runs_c = {
        "m": {
            "fmt": resp("fmt", retrieved=(A,)),
            "empty": resp("empty", retrieved=(B,)),
            "none": resp("none", retrieved=("mq-" + "e" * 16,)),
        }
    }
    out = cmp.descriptive_topic_analyses({}, c_ex, topic_of, {}, runs_c)["i_hard_negative_on_topic"]["m"]
    assert out["on_topic"]["n"] == 1  # only 'fmt'
    assert out["off_topic"]["n"] == 2  # 'empty' and 'none' never count as on-topic


def test_copy_rate_uses_supplied_chunk_text_not_whole_record() -> None:
    a_ex = {"a1": ex("a1"), "a2": ex("a2")}
    chunk = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    rs = {
        "a1": resp("a1", retrieved=(A,), answer=f"alpha beta gamma delta epsilon zeta eta theta [{A}]."),
        "a2": resp("a2", retrieved=(A, B), answer="lambda mu nu xi omicron pi rho sigma tau."),
    }

    def retrieve(q: str) -> tuple[list[tuple[str, str]], list[str]]:
        return ([(A, chunk)], []) if q.endswith("a1?") else ([(B, chunk)], [])  # a2: ids differ -> mismatch

    supplied, mismatch = cmp.supplied_chunk_texts(rs, a_ex, retrieve)
    assert supplied == {"a1": [chunk]} and mismatch == 1
    risk = cmp.risk_metrics(a_ex, {"m": rs}, {"m": supplied}, {"m": mismatch})["m"]
    assert risk["copy_rate_mean_answered"] == 1.0 and risk["copy_rate_n"] == 1
    assert risk["copy_rate_evidence_mismatch_excluded"] == 1
