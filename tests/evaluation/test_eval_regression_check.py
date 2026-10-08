"""Offline tests for the f4bf7e9 §3 regression rules (synthetic responses only)."""

from __future__ import annotations

from typing import Any

from medquad_qa.contracts import EvaluationExample, QAResponse
from medquad_qa.evaluation import regression_check as rc


def ex(eid: str, **kw: Any) -> EvaluationExample:
    base: dict[str, Any] = dict(
        example_id=eid,
        question=f"Synthetic {eid}?",
        answerable=True,
        label_provenance="synthetic_rule",
        question_provenance="synthetic_rule",
        eval_split="test",
        evaluation_track="answerability_robustness",
        split_group_id=f"g-{eid}",
    )
    base.update(kw)
    return EvaluationExample(**base)


def resp(eid: str, abstained: bool, reason: Any = None) -> QAResponse:
    return QAResponse(
        request_id=eid,
        experiment_mode="rag",
        answer="a",
        abstained=abstained,
        abstention_reason=reason,
        model_version="m",
        prompt_version="p",
        latency_ms=1.0,
    )


def test_track_a_non_inferiority() -> None:
    a = {f"a{i}": ex(f"a{i}") for i in range(100)}
    old = {i: resp(i, False) for i in a}
    same = rc.track_a("base", a, old, dict(old), {}, {}, None)
    assert same["answered_rate"]["regression"] is False
    worse = {i: resp(i, k < 20, "insufficient_evidence") for k, i in enumerate(a)}  # 20 points fewer answered
    assert rc.track_a("base", a, old, worse, {}, {}, None)["answered_rate"]["regression"] is True
    rag = rc.track_a(
        "rag", a, old, dict(old), {}, {}, {"citation_validity": 0.98, "responses_with_invariant_violation": 0}
    )
    assert rag["citation_validity"]["regression"] is True


def test_track_c_l40_and_strict_personal() -> None:
    c = {
        f"h{i}": ex(
            f"h{i}",
            answerable=False,
            expected_behavior="abstain",
            case_type="unanswerable",
            expected_abstention_reason="insufficient_evidence",
            notes="hard_negative:x",
        )
        for i in range(50)
    }
    c |= {
        f"p{i}": ex(
            f"p{i}",
            answerable=False,
            expected_behavior="abstain",
            case_type="personalized_advice",
            expected_abstention_reason="personalized_medical_advice",
            notes="personalized_advice:x",
        )
        for i in range(40)
    }
    old = {
        i: resp(
            i, True, "personalized_medical_advice" if i.startswith("p") and int(i[1:]) < 30 else "insufficient_evidence"
        )
        for i in c
    }
    # small drop in hard-negative recall (2/50): L40 does NOT flag (CI upper not below -0.05)
    new = dict(old)
    for i in ("h0", "h1"):
        new[i] = resp(i, False)
    out = rc.track_c("rag", c, old, new)
    assert out["abstain_recall_hard_negative"]["regression"] is False
    assert out["personal_refusals_seen"] == {**out["personal_refusals_seen"], "e4": 30, "new": 30, "regression": False}
    new["p0"] = resp("p0", False)
    assert rc.track_c("rag", c, old, new)["personal_refusals_seen"]["regression"] is True  # 29 < 30
    assert rc.any_regression({"x": rc.track_c("rag", c, old, new)}) is True
