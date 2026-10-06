"""Offline tests for the evaluation metric harness (synthetic fixtures only; no MedQuAD text)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from medquad_qa.contracts.qa import Citation, QAResponse
from medquad_qa.evaluation import (
    abstention,
    classifier_metrics,
    latency,
    retrieval_metrics,
    rubric,
    stats,
    text_metrics,
)
from medquad_qa.evaluation.citation_metrics import aggregate as agg_citations
from medquad_qa.evaluation.citation_metrics import score_response
from medquad_qa.evaluation.claims import extract_claims
from medquad_qa.evaluation.run_format import load_run
from medquad_qa.evaluation.support import LexicalSupportScorer

A, B, C, D = (f"mq-{c * 16}" for c in "abcd")


# ------------------------------------------------------------------ retrieval
def test_hit_mrr_and_gold_recall() -> None:
    ranked = [C, A, D, B]
    s = retrieval_metrics.score_query("q1", ranked, {A, B}, ks=(1, 2, 4))
    assert s.hit_at == {1: 0.0, 2: 1.0, 4: 1.0}
    assert s.gold_recall_at == {1: 0.0, 2: 0.5, 4: 1.0}
    assert s.reciprocal_rank == 0.5 and s.first_relevant_rank == 2


def test_mrr_respects_cutoff_and_empty_gold_rejected() -> None:
    s = retrieval_metrics.score_query("q", [C, D, A], {A}, ks=(1, 2))
    assert s.reciprocal_rank == 0.0 and s.first_relevant_rank is None
    with pytest.raises(ValueError):
        retrieval_metrics.score_query("q", [A], set())


def test_ndcg_requires_grades_and_matches_hand_value() -> None:
    with pytest.raises(ValueError):
        retrieval_metrics.ndcg_at_k([A], {}, 1)
    v = retrieval_metrics.ndcg_at_k([B, A], {A: 2, B: 1}, 2)
    dcg = 1 / math.log2(2) + 3 / math.log2(3)
    idcg = 3 / math.log2(2) + 1 / math.log2(3)
    assert v == pytest.approx(dcg / idcg)


def test_aggregate_macro() -> None:
    s1 = retrieval_metrics.score_query("q1", [A], {A}, ks=(1,))
    s2 = retrieval_metrics.score_query("q2", [C, A], {A}, ks=(1,))
    agg = retrieval_metrics.aggregate([s1, s2])
    assert agg["recall@1"] == 0.5 and agg["mrr@1"] == 0.5 and agg["n"] == 2


# ------------------------------------------------------------------ run format
def _run_line(eid: str, hits: list[dict[str, object]], **kw: object) -> str:
    row = {
        "schema": "medquad-retrieval-run-v1",
        "example_id": eid,
        "retriever": "bm25:answer",
        "corpus_version": "c",
        "top_k": 3,
        "latency_ms": 1.0,
    }
    row.update(kw)
    row["hits"] = hits
    return json.dumps(row)


def test_run_file_valid_and_invalid(tmp_path: Path) -> None:
    ok = tmp_path / "ok.jsonl"
    ok.write_text(
        _run_line("e1", [{"record_id": A, "rank": 1, "score": 2.0}, {"record_id": B, "rank": 2, "score": 1.0}])
        + "\n"
        + _run_line("e2", [])
        + "\n"
    )
    rows = load_run(ok, expected_example_ids={"e1", "e2"})
    assert rows[0].ranked_ids() == [A, B] and rows[1].hits == []
    with pytest.raises(ValueError, match="mismatch"):
        load_run(ok, expected_example_ids={"e1"})
    for bad_hits in (
        [{"record_id": A, "rank": 2, "score": 1.0}],  # rank gap
        [{"record_id": A, "rank": 1, "score": 1.0}, {"record_id": A, "rank": 2, "score": 0.5}],  # duplicate id
    ):
        bad = tmp_path / "bad.jsonl"
        bad.write_text(_run_line("e1", bad_hits) + "\n")
        with pytest.raises(ValueError):
            load_run(bad)
    wrong_schema = tmp_path / "schema.jsonl"
    wrong_schema.write_text(_run_line("e1", [], schema="other-v9") + "\n")
    with pytest.raises(ValueError):
        load_run(wrong_schema)
    leak = tmp_path / "leak.jsonl"
    leak.write_text(_run_line("e1", [], question="text must not appear") + "\n")
    with pytest.raises(ValueError):
        load_run(leak)


# ------------------------------------------------------------------ claims + citations
def test_extract_claims_attaches_trailing_markers() -> None:
    text = (
        f"Glaucoma damages the optic nerve [{A}]. It can cause vision loss over time. [{B}] Ok. INSUFFICIENT_EVIDENCE"
    )
    claims = extract_claims(text)
    assert [c.cited_ids for c in claims] == [(A,), (B,)]
    assert claims[0].text == "Glaucoma damages the optic nerve."


def _resp(answer: str, *, cited: list[str], retrieved: list[str], invalid: list[str] | None = None) -> QAResponse:
    return QAResponse(
        request_id="r1",
        experiment_mode="rag",
        answer=answer,
        abstained=False,
        citations=[Citation(record_id=r, evidence_snippet="x") for r in cited],
        retrieved_record_ids=retrieved,
        invalid_citation_ids=invalid or [],
        model_version="m",
        prompt_version="p",
        latency_ms=10.0,
    )


def test_citation_scores_with_lexical_scorer() -> None:
    evidence = {
        A: "Synthetic condition X causes persistent itching of the elbows.",
        B: "Synthetic condition X is treated with rest and fluids.",
    }
    answer = (
        f"Condition X causes persistent itching of the elbows [{A}]. "
        f"Condition X is cured by surgery on the knees [{B}]. "
        "Patients often recover fully within days."
    )
    s = score_response(
        _resp(answer, cited=[A, B], retrieved=[A, B], invalid=["E9"]), evidence, LexicalSupportScorer(), 0.8
    )
    assert (s.n_claims, s.n_cited_claims, s.n_supported_cited_claims) == (3, 2, 1)
    assert s.validity == pytest.approx(2 / 3)
    assert s.coverage == pytest.approx(2 / 3)
    assert s.support_rate == 0.5
    assert s.n_unsupported_claims == 2 and s.invariant_violations == []
    agg = agg_citations([s])
    assert agg["citation_support"] == 0.5 and agg["unsupported_claim_rate"] == pytest.approx(2 / 3)


def test_citation_invariant_violation_and_leaked_labels() -> None:
    s = score_response(
        _resp(f"Condition X causes itching of elbows [{C}] and more [E2].", cited=[A], retrieved=[A]),
        {A: "Condition X causes itching of elbows."},
        LexicalSupportScorer(),
    )
    assert s.invariant_violations == [C] and s.leaked_labels == 1


def test_abstained_response_not_scored() -> None:
    r = _resp("I cannot answer.", cited=[], retrieved=[]).model_copy(update={"abstained": True})
    with pytest.raises(ValueError):
        score_response(r, {}, LexicalSupportScorer())


def test_lexical_scorer_keeps_negations() -> None:
    sc = LexicalSupportScorer()
    assert sc.score("X is not contagious.", "X is not contagious.") == 1.0
    assert sc.score("X is contagious.", "X is not contagious.") < 1.0


# ------------------------------------------------------------------ abstention
def test_abstention_confusion() -> None:
    Out = abstention.AbstentionOutcome
    outs = [
        Out("1", "abstain", True, "personalized_medical_advice", "personalized_medical_advice", "personalized_advice"),
        Out("2", "abstain", True, "no_relevant_evidence", "insufficient_evidence", "unanswerable"),
        Out("3", "abstain", False, group="unanswerable"),
        Out("4", "answer", True, group="answerable"),
        Out("5", "answer", False, group="answerable"),
        Out("6", "either", True, group="ambiguous"),
    ]
    c = abstention.confusion(outs)
    assert (c["tp"], c["fn"], c["fp"], c["tn"], c["n_either"]) == (2, 1, 1, 1, 1)
    assert c["abstain_precision"] == pytest.approx(2 / 3)
    assert c["abstain_recall"] == pytest.approx(2 / 3)
    assert c["over_refusal_rate"] == 0.5
    assert c["reason_accuracy"] == 0.5
    by = abstention.confusion_by_group(outs)
    assert by["answerable"]["over_refusal_rate"] == 0.5 and "all" in by


# ------------------------------------------------------------------ classifier
def test_classifier_report_and_ece() -> None:
    y = [0, 0, 1, 1, 1, 0]
    p = [0.1, 0.4, 0.35, 0.8, 0.95, 0.05]
    r = classifier_metrics.classifier_report(y, p, threshold=0.5)
    assert r["auroc"] == pytest.approx(roc_auc_score(y, p))
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (2, 0, 1, 3)
    assert r["brier"] == pytest.approx(np.mean((np.array(p) - np.array(y)) ** 2))
    # perfectly calibrated toy: ECE 0
    assert classifier_metrics.expected_calibration_error([1, 0], [1.0, 0.0]) == 0.0
    # hand ECE: bins [0.3,0.4): {0.35->1} gap .65; [0.4,0.5): {0.4->0} .4; [0.0,0.1): {0.05->0} .05;
    # [0.1,0.2): {0.1->0} .1; [0.8,0.9): {0.8->1} .2; [0.9,1]: {0.95->1} .05
    assert r["ece"] == pytest.approx((0.65 + 0.4 + 0.05 + 0.1 + 0.2 + 0.05) / 6)


def test_classifier_single_class_and_bad_input() -> None:
    r = classifier_metrics.classifier_report([1, 1], [0.6, 0.9], threshold=0.5)
    assert r["auroc"] is None and r["recall"] == 1.0
    with pytest.raises(ValueError):
        classifier_metrics.classifier_report([0, 1], [0.2, 1.2], threshold=0.5)


# ------------------------------------------------------------------ latency / text / rubric
def test_latency_summary() -> None:
    s = latency.summarize(list(range(1, 101)))
    assert s["median_ms"] == 50.5 and s["p95_ms"] == pytest.approx(95.05)
    comp = latency.summarize_components([{"retrieval": 1.0, "generation": 5.0}, {"retrieval": 3.0}])
    assert comp["retrieval"]["n"] == 2 and comp["generation"]["n"] == 1


def test_text_metrics_ignore_citation_markers() -> None:
    assert text_metrics.token_f1(f"optic nerve damage [{A}]", "optic nerve damage") == 1.0
    assert text_metrics.rouge_l_f(f"optic nerve damage [{A}]", "optic nerve damage") == 1.0
    assert text_metrics.token_f1("", "x") == 0.0


def test_rubric_grader_labels_and_agreement() -> None:
    rubric.RubricScore(example_id="e", experiment_mode="rag", correctness=2, completeness=1, grader="ai_agent")
    for bad in ("human", "clinician", "doctor"):
        with pytest.raises(ValueError):
            rubric.RubricScore(example_id="e", experiment_mode="rag", correctness=2, completeness=1, grader=bad)
    ag = rubric.agreement([2, 1, 0, 2], [2, 1, 1, 2])
    assert ag["exact_agreement"] == 0.75 and ag["weighted_kappa"] is not None


# ------------------------------------------------------------------ statistics
def test_bootstrap_detects_clear_difference_and_is_seeded() -> None:
    rng = np.random.default_rng(0)
    a = rng.random(200).round().tolist()
    b = [1.0] * 200
    clusters = [f"g{i // 2}" for i in range(200)]
    r1 = stats.paired_cluster_bootstrap(a, b, clusters, n_resamples=2000)
    r2 = stats.paired_cluster_bootstrap(a, b, clusters, n_resamples=2000)
    assert r1 == r2 and r1["n_clusters"] == 100
    assert r1["ci_low"] > 0 and r1["p_value"] <= 1 / 2001 + 1e-12


def test_bootstrap_null_difference() -> None:
    r = stats.paired_cluster_bootstrap([1, 0, 1, 0], [1, 0, 1, 0], ["a", "b", "c", "d"], n_resamples=500)
    assert r["diff_b_minus_a"] == 0.0 and r["p_value"] == 1.0


def test_exact_mcnemar() -> None:
    a = [True] * 10 + [False] * 10
    b = [True] * 10 + [True] * 10
    r = stats.exact_mcnemar(a, b)
    assert (r["a_only_correct"], r["b_only_correct"]) == (0, 10)
    assert r["p_value"] == pytest.approx(2 * 0.5**10)
    assert stats.exact_mcnemar([True], [True])["p_value"] == 1.0
