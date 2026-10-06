"""Answerability pairs, lexical LR baseline, serving predictor and metrics on synthetic records."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from training_fixtures import synthetic_record

from medquad_qa.contracts.interfaces import AnswerabilityPredictor, ArtifactUnavailableError
from medquad_qa.contracts.records import MedicalRecord
from medquad_qa.models.answerability import LexicalAnswerabilityPredictor
from medquad_qa.training import classifier_metrics as cm
from medquad_qa.training.answerability_pairs import PAIR_RULE_VERSION, build_pairs, read_pairs, write_pairs
from medquad_qa.training.answerability_train import train_lexical_baseline

WORDS = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "kappa", "theta", "lambda", "zeta"]
ASPECT = {"symptoms": "signs include fever cough pain", "treatment": "treated with rest fluids medicine"}


def _records(prefix: str, n_topics: int, start: int) -> list[MedicalRecord]:
    out, i = [], start
    for k in range(n_topics):
        topic = f"{WORDS[k % 10]}{k}{prefix}"
        for qtype in ("symptoms", "treatment"):
            answer = f"{topic} disease. The {qtype} of {topic}: {ASPECT[qtype]}. More text about {topic} here."
            flags = ["boilerplate_answer"] if k == 0 and qtype == "treatment" else []
            out.append(synthetic_record(i, topic, qtype, answer, group=f"g-{prefix}-{k}", flags=flags))
            i += 1
    return out


def test_pairs_follow_rules(tmp_path: Path) -> None:
    recs = _records("tr", 12, 0)
    pairs = build_pairs(recs, "train")
    by_id = {r.record_id: r for r in recs}
    assert pairs == build_pairs(recs, "train")  # deterministic
    boiler = {r.record_id for r in recs if r.quality_flags}
    assert not boiler & {p.evidence_record_id for p in pairs}
    assert not boiler & {p.question_record_id for p in pairs}
    for p in pairs:
        q, e = by_id[p.question_record_id], by_id[p.evidence_record_id]
        assert p.pair_rule_version == PAIR_RULE_VERSION and p.split == "train"
        assert p.split_group_id == q.split_group_id and p.label_provenance == "synthetic_rule"
        if p.label:
            assert p.negative_type is None and (q.topic, q.question_type) == (e.topic, e.question_type)
        elif p.negative_type == "same_topic_diff_qtype":
            assert q.topic == e.topic and q.question_type != e.question_type
        else:
            assert q.topic != e.topic and q.split_group_id != e.split_group_id  # C1
    n_pos = sum(p.label for p in pairs)
    assert n_pos == len(pairs) - n_pos  # 1:1
    assert {p.negative_type for p in pairs if not p.label} <= {
        "easy_random",
        "same_topic_diff_qtype",
        "lexical_hard_bm25",
    }

    write_pairs(pairs, tmp_path / "pairs.jsonl", tmp_path / "manifest.jsonl", split_version="split-x")
    assert read_pairs(tmp_path / "pairs.jsonl") == pairs
    rows = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert "question" not in rows[0] and "evidence_text" not in rows[0]
    assert {r["split_version"] for r in rows} == {"split-x"}
    assert sum(r["own_answer"] for r in rows) == len({p.question_record_id for p in pairs})  # C4


def test_c2_drops_negative_identical_to_positive() -> None:
    recs = _records("tr", 6, 0)
    # different topic and groups, but the same answer text as record 0 (up to whitespace)
    clone = recs[0].model_copy(
        update={
            "record_id": "mq-" + "f" * 16,
            "topic": "zzz",
            "split_group_id": "g-other",
            "duplicate_group_id": "d-x",
            "answer": "  " + recs[0].answer.replace(" ", "  "),
        }
    )
    stats: dict[str, int] = {}
    pairs = build_pairs([*recs, clone], "train", stats=stats)
    assert not [
        p
        for p in pairs
        if p.question_record_id == recs[0].record_id and not p.label and p.evidence_record_id == clone.record_id
    ]
    assert not [
        p
        for p in pairs
        if p.question_record_id == clone.record_id and not p.label and p.evidence_record_id == recs[0].record_id
    ]
    assert "c2_dropped_negatives" in stats


def test_metrics_and_threshold() -> None:
    y = np.array([0, 0, 1, 1, 1, 0])
    p = np.array([0.1, 0.4, 0.35, 0.8, 0.9, 0.2])
    thr, f1 = cm.max_f1_threshold(y, p)
    assert thr == pytest.approx(0.35) and f1 == pytest.approx(0.857142857, rel=1e-6)
    ece, bins = cm.reliability(y, p)
    assert len(bins) == 10 and sum(b["n"] for b in bins) == 6 and 0 <= ece <= 1
    block = cm.evaluate(y, p, thr, [None, "easy_random", None, None, None, "easy_random"])
    assert block["at_threshold"]["tp"] == 3 and block["by_type"]["easy_random"]["n"] == 2


def test_lexical_baseline_trains_and_serves(tmp_path: Path) -> None:
    pairs = {
        "train": build_pairs(_records("tr", 30, 0), "train"),
        "validation": build_pairs(_records("va", 10, 1000), "validation"),
        "test": build_pairs(_records("te", 10, 2000), "test"),
    }
    out = tmp_path / "lexlr"
    result = train_lexical_baseline(pairs, out)
    assert result["val"]["roc_auc"] > 0.5 and "test" in result and "test_excl_own_answer" in result
    assert result["threshold"]["threshold_rule"] == "max_f1_on_validation"

    pred = LexicalAnswerabilityPredictor.load(out)
    assert isinstance(pred, AnswerabilityPredictor)
    assert pred.model_version == result["model_version"]
    empty = pred.predict("What are the symptoms of x?", [])
    assert (empty.answerability_score, empty.predicted_label, empty.per_evidence_scores) == (0.0, False, [])
    q = pairs["validation"][0].question
    out_pred = pred.predict(q, ["unrelated words entirely", pairs["validation"][0].evidence_text])
    assert out_pred.answerability_score == max(out_pred.per_evidence_scores)
    assert out_pred.predicted_label == (out_pred.answerability_score >= pred.threshold)
    assert out_pred.threshold_version == result["threshold"]["threshold_version"]


def test_predictor_missing_files(tmp_path: Path) -> None:
    with pytest.raises(ArtifactUnavailableError):
        LexicalAnswerabilityPredictor.load(tmp_path)
