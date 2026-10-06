"""Train the lexical logistic-regression answerability baseline and evaluate it with a validation threshold."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from medquad_qa.contracts.answerability import AnswerabilityPair
from medquad_qa.models.answerability import FEATURE_NAMES, LexicalFeaturizer, LexicalLRModel, lexlr_version
from medquad_qa.training import classifier_metrics as cm
from medquad_qa.training.answerability_pairs import is_own_answer


def labels(pairs: list[AnswerabilityPair]) -> np.ndarray:
    return np.asarray([int(p.label) for p in pairs])


def pair_stats(pairs: list[AnswerabilityPair]) -> dict[str, Any]:
    neg: dict[str, int] = {}
    for p in pairs:
        if not p.label:
            neg[str(p.negative_type)] = neg.get(str(p.negative_type), 0) + 1
    return {
        "n": len(pairs),
        "n_pos": int(sum(p.label for p in pairs)),
        "n_pos_self": sum(is_own_answer(p) for p in pairs),
        "n_pos_sib": sum(":pos-sib:" in p.pair_id for p in pairs),
        "negatives_by_type": neg,
        "n_questions": len({p.question_record_id for p in pairs}),
        "n_split_groups": len({p.split_group_id for p in pairs}),
    }


def eval_blocks(prefix: str, pairs: list[AnswerabilityPair], p: np.ndarray, threshold: float) -> dict[str, Any]:
    """Metrics on all pairs and, as the headline (evaluator C4), on pairs without own-answer positives."""
    keep = np.asarray([not is_own_answer(x) for x in pairs])
    sub = [x for x, k in zip(pairs, keep, strict=True) if k]
    return {
        prefix: cm.evaluate(labels(pairs), p, threshold, [x.negative_type for x in pairs]),
        f"{prefix}_excl_own_answer": cm.evaluate(labels(sub), p[keep], threshold, [x.negative_type for x in sub]),
    }


def threshold_block(y_val: np.ndarray, p_val: np.ndarray, model_version: str) -> dict[str, Any]:
    thr, f1 = cm.max_f1_threshold(y_val, p_val)
    return {
        "threshold": thr,
        "threshold_rule": "max_f1_on_validation",
        "val_f1_at_threshold": f1,
        "precision_0.90_threshold": cm.precision_target_threshold(y_val, p_val, 0.90),
        "threshold_version": f"{model_version}:maxf1-val:{thr:.6f}",
    }


def train_lexical_baseline(
    pairs: dict[str, list[AnswerabilityPair]], out_dir: Path, seed: int = 42, c: float = 1.0
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression

    train, val, test = pairs["train"], pairs["validation"], pairs.get("test", [])
    featurizer = LexicalFeaturizer.fit({p.evidence_record_id: p.evidence_text for p in train}.values())
    x_tr = featurizer.matrix([p.question for p in train], [p.evidence_text for p in train])
    mean, std = x_tr.mean(axis=0), x_tr.std(axis=0)
    std[std == 0] = 1.0
    clf = LogisticRegression(C=c, max_iter=2000, random_state=seed)
    clf.fit((x_tr - mean) / std, labels(train))
    model = LexicalLRModel(featurizer, mean, std, clf.coef_[0], float(clf.intercept_[0]))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model.json").write_text(model.to_json(), encoding="utf-8")
    p_val = model.score([p.question for p in val], [p.evidence_text for p in val])
    model_version = lexlr_version((out_dir / "model.json").read_bytes())  # same rule as the serving predictor
    thr = threshold_block(labels(val), p_val, model_version)
    (out_dir / "threshold.json").write_text(
        json.dumps({k: thr[k] for k in ("threshold", "threshold_version", "threshold_rule")}, indent=2) + "\n",
        encoding="utf-8",
    )

    result: dict[str, Any] = {
        "model": "lexical_lr",
        "model_version": model_version,
        "features": list(FEATURE_NAMES),
        "coefficients": dict(zip(FEATURE_NAMES, map(float, clf.coef_[0]), strict=True)),
        "hyperparameters": {"C": c, "solver": clf.solver, "max_iter": 2000, "seed": seed, "standardised": True},
        "threshold": thr,
        **eval_blocks("val", val, p_val, thr["threshold"]),
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if test:
        p_test = model.score([p.question for p in test], [p.evidence_text for p in test])
        result.update(eval_blocks("test", test, p_test, thr["threshold"]))
        np.save(out_dir / "test_scores.npy", p_test)
    np.save(out_dir / "val_scores.npy", p_val)
    return result
