"""Leakage checks, frozen eval-set I/O, per-mode QA summary and FixtureRetriever (synthetic fixtures only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from medquad_qa.contracts.evaluation import EvaluationExample
from medquad_qa.contracts.interfaces import Retriever
from medquad_qa.contracts.qa import GenerationMeta, GenerationParams, QAResponse
from medquad_qa.evaluation.evalset import load_eval_set, write_eval_set
from medquad_qa.evaluation.fixture_retriever import FixtureRetriever, fixture_record_id
from medquad_qa.evaluation.leakage import LeakageError, check_leakage, normalize_question
from medquad_qa.evaluation.qa_report import summarize_mode

TR, VA, TE, TE2 = (f"mq-{c * 16}" for c in "1234")
RECORD_GROUP = {TR: "g-train", VA: "g-val", TE: "g-test", TE2: "g-test2"}
GROUP_SPLIT: dict[str, Any] = {"g-train": "train", "g-val": "validation", "g-test": "test", "g-test2": "test"}
INDEXED = ["What is synthetic condition X ?", "How to treat synthetic condition Y?"]


def ex(eid: str = "t1", **kw: Any) -> EvaluationExample:
    base: dict[str, Any] = dict(
        example_id=eid,
        question="Could you explain what synthetic disorder X involves?",
        gold_record_ids=[TE],
        derived_from_record_ids=[TE],
        split_group_id="g-test",
        answerable=True,
        label_provenance="synthetic_rule",
        question_provenance="synthetic_rule",
        eval_split="test",
        evaluation_track="corpus_grounded",
    )
    base.update(kw)
    return EvaluationExample(**base)


def run(examples: list[EvaluationExample], **kw: Any) -> Any:
    return check_leakage(examples, RECORD_GROUP, GROUP_SPLIT, INDEXED, **kw)


def test_clean_set_passes() -> None:
    rep = run(
        [
            ex(),
            ex(
                "d1",
                question="Tell me about Y please, in brief",
                gold_record_ids=[VA],
                derived_from_record_ids=[VA],
                split_group_id="g-val",
                eval_split="dev",
            ),
        ]
    )
    assert rep.passed and rep.counts_by_eval_split == {"test": 1, "dev": 1}


@pytest.mark.parametrize(
    "examples,code",
    [
        ([ex(), ex()], "E1"),
        ([ex(), ex("t2")], "E1"),  # same normalized question
        ([ex(gold_record_ids=[TR])], "E2"),
        ([ex(eval_split="dev")], "E2"),
        ([ex(question="what is SYNTHETIC condition x")], "E3"),
        ([ex(gold_record_ids=["mq-" + "9" * 16])], "E5"),
    ],
)
def test_hard_failures(examples: list[EvaluationExample], code: str) -> None:
    with pytest.raises(LeakageError, match=code):
        run(examples)
    assert any(e.startswith(code) for e in run(examples, fail=False).errors)


def test_protected_overlap() -> None:
    with pytest.raises(LeakageError, match="SFT"):
        run([ex()], sft_record_ids=[TE])
    with pytest.raises(LeakageError, match="classifier training"):
        run([ex()], classifier_train_record_ids=[TE])
    with pytest.raises(LeakageError, match="threshold"):
        run([ex()], threshold_record_ids=[TE2, TE])
    # dev items may overlap threshold-fitting (validation) data
    dev = ex("d1", gold_record_ids=[VA], derived_from_record_ids=[VA], split_group_id="g-val", eval_split="dev")
    assert run([dev], threshold_record_ids=[VA]).passed


def test_train_probe_rules() -> None:
    probe = ex(
        "p1",
        question=INDEXED[0],
        gold_record_ids=[TR],
        derived_from_record_ids=[TR],
        split_group_id="g-train",
        eval_split="train_probe",
        evaluation_track="finetune_generalization",
    )
    rep = run([probe], sft_record_ids=[TR])
    assert rep.passed and rep.train_probe_sft_overlap == 1 and rep.warnings
    with pytest.raises(LeakageError, match="E2"):
        run([probe.model_copy(update={"gold_record_ids": [TE]})])


def test_near_copy_is_warning() -> None:
    with pytest.raises(LeakageError, match="E3"):  # differs only by '?': normalized-equal, a hard error
        run([ex(question="How to treat synthetic condition Y")])
    rep = run([ex(question="How to treat the synthetic condition Y?")], near_copy_jaccard=0.8)
    assert rep.passed and any("near-copy" in w for w in rep.warnings)


def test_normalize_question() -> None:
    assert normalize_question("  What IS  X ? ") == normalize_question("what is x")


# ------------------------------------------------------------------ eval set I/O
def test_write_load_roundtrip_and_tamper(tmp_path: Path) -> None:
    p = tmp_path / "evalset.jsonl"
    man = write_eval_set([ex()], p, meta={"split_version": "split-test"})
    assert man["n_examples"] == 1 and man["split_version"] == "split-test"
    assert load_eval_set(p)[0] == ex()
    p.write_text(p.read_text().replace("synthetic disorder", "synthetic disease"))
    with pytest.raises(ValueError, match="sha256"):
        load_eval_set(p)


@pytest.mark.parametrize(
    "kw",
    [
        {"reference_answer": "dataset text"},
        {"label_provenance": "human_reviewed"},  # human provenance without human reviewer
        {"reviewer": "human_nonclinical"},  # human reviewer without human provenance
        {"expected_behavior": "abstain", "expected_abstention_reason": "no_relevant_evidence"},
        {"answerable": False, "expected_behavior": "abstain"},  # missing reason
    ],
)
def test_policy_rejections(tmp_path: Path, kw: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        write_eval_set([ex(**kw)], tmp_path / "x.jsonl")


# ------------------------------------------------------------------ per-mode summary
def _r(
    eid: str,
    *,
    abstained: bool,
    reason: Any = None,
    warnings: list[str] | None = None,
    retrieved: list[str] | None = None,
    mode: Any = "rag",
    prompt: str = "p1",
) -> QAResponse:
    return QAResponse(
        request_id=eid,
        experiment_mode=mode,
        answer="a",
        abstained=abstained,
        abstention_reason=reason,
        retrieved_record_ids=retrieved or [],
        warnings=warnings or [],
        model_version="m",
        prompt_version=prompt,
        latency_ms=100.0,
        component_latency_ms={"retrieval": 5.0},
        generation=None
        if abstained
        else GenerationMeta(params=GenerationParams(), prompt_tokens=50, completion_tokens=20, finish_reason="stop"),
    )


def test_summarize_mode() -> None:
    examples = [
        ex("a1"),
        ex(
            "u1",
            question="q u1 synthetic",
            answerable=False,
            expected_behavior="abstain",
            expected_abstention_reason="no_relevant_evidence",
            case_type="unanswerable",
            notes="hard_negative:bm25",
            gold_record_ids=[],
        ),
    ]
    responses = [
        _r("a1", abstained=False, warnings=["evidence_truncated"], retrieved=["mq-" + "0" * 16, TE]),
        _r("u1", abstained=True, reason="no_relevant_evidence"),
    ]
    s = summarize_mode(examples, responses)
    assert s["evidence_truncated_rate"] == 0.5
    assert s["abstention"]["all"]["tp"] == 1 and s["abstention"]["all"]["tn"] == 1
    assert "unanswerable:hard_negative" in s["abstention"]
    assert s["generation"]["matched_params"] is True
    assert s["supplied_evidence_retrieval"]["recall@1"] == 0.0
    assert s["supplied_evidence_retrieval"]["recall@3"] == 1.0
    with pytest.raises(ValueError, match="prompt_version"):
        summarize_mode(examples, [responses[0], _r("u1", abstained=True, prompt="p2")])
    with pytest.raises(ValueError, match="mixed"):
        summarize_mode(examples, [responses[0], _r("u1", abstained=True, mode="base")])


# ------------------------------------------------------------------ fixture retriever
def test_fixture_retriever_protocol() -> None:
    fr = FixtureRetriever({"Is X contagious?": ["X is contagious.", "X is not contagious."]})
    assert isinstance(fr, Retriever)
    hits = fr.retrieve("  Is X contagious?  ", 5)
    assert [h.rank for h in hits] == [1, 2]
    assert hits[1].record_id == fixture_record_id("X is not contagious.")
    assert fr.retrieve("unknown", 5) == [] and len(fr.retrieve("Is X contagious?", 1)) == 1
    with pytest.raises(ValueError):
        fr.retrieve("Is X contagious?", 0)


def test_closed_book_expectations_follow_d023() -> None:
    from medquad_qa.evaluation.qa_report import expected_for_mode

    unans = ex(
        "u",
        answerable=False,
        expected_behavior="abstain",
        expected_abstention_reason="no_relevant_evidence",
        case_type="unanswerable",
    )
    personal = ex(
        "p",
        answerable=False,
        expected_behavior="abstain",
        expected_abstention_reason="personalized_medical_advice",
        case_type="personalized_advice",
    )
    assert expected_for_mode(unans, "rag") == "abstain" and expected_for_mode(unans, "base") == "either"
    assert expected_for_mode(personal, "base") == "abstain" and expected_for_mode(personal, "finetuned") == "abstain"
    assert expected_for_mode(ex(), "finetuned") == "answer"
