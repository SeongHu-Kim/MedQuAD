"""Answerability-classifier discrimination and calibration (non-clinical).

ECE uses 10 equal-width bins on [0, 1] (the last bin closed on the right), weighted by bin size, matching the
model-engineer's definition so numbers are comparable. Thresholded metrics use the frozen threshold passed in;
this module never picks a threshold on the data it scores.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _arrays(y_true: Sequence[int | bool], y_score: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_score, dtype=float)
    if y.shape != p.shape or y.ndim != 1 or y.size == 0:
        raise ValueError("y_true and y_score must be equal-length non-empty 1-D sequences")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y_true must be binary")
    if ((p < 0) | (p > 1)).any() or np.isnan(p).any():
        raise ValueError("y_score must be probabilities in [0, 1]")
    return y, p


def reliability_bins(
    y_true: Sequence[int | bool], y_score: Sequence[float], n_bins: int = 10
) -> list[dict[str, float]]:
    y, p = _arrays(y_true, y_score)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    bins = []
    for b in range(n_bins):
        m = idx == b
        bins.append(
            {
                "lo": b / n_bins,
                "hi": (b + 1) / n_bins,
                "count": int(m.sum()),
                "mean_score": float(p[m].mean()) if m.any() else float("nan"),
                "frac_positive": float(y[m].mean()) if m.any() else float("nan"),
            }
        )
    return bins


def expected_calibration_error(y_true: Sequence[int | bool], y_score: Sequence[float], n_bins: int = 10) -> float:
    n = len(y_true)
    return float(
        sum(
            b["count"] / n * abs(b["frac_positive"] - b["mean_score"])
            for b in reliability_bins(y_true, y_score, n_bins)
            if b["count"]
        )
    )


def classifier_report(
    y_true: Sequence[int | bool], y_score: Sequence[float], threshold: float, n_bins: int = 10
) -> dict[str, float | int | None]:
    y, p = _arrays(y_true, y_score)
    both = len(set(y.tolist())) == 2
    pred = p >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    return {
        "n": int(y.size),
        "n_positive": int(y.sum()),
        "auroc": float(roc_auc_score(y, p)) if both else None,
        "auprc": float(average_precision_score(y, p)) if both else None,
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y_true, y_score, n_bins),
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec and rec else None,
        "accuracy": (tp + tn) / int(y.size),
    }
