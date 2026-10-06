"""Discrimination and calibration metrics for the answerability classifiers; threshold chosen on validation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score

N_BINS = 10


def max_f1_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Threshold maximising F1 on (y, p). Returns (threshold, f1)."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    i = int(np.argmax(f1))
    return float(thr[i]), float(f1[i])


def precision_target_threshold(y: np.ndarray, p: np.ndarray, target: float = 0.90) -> float | None:
    """Lowest threshold whose precision >= target (secondary operating point, D-018)."""
    prec, _, thr = precision_recall_curve(y, p)
    ok = np.where(prec[:-1] >= target)[0]
    return float(thr[ok[0]]) if len(ok) else None


def reliability(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> tuple[float, list[dict[str, Any]]]:
    """ECE over equal-width bins, plus per-bin (count, mean predicted, observed positive rate)."""
    # same binning as medquad_qa.evaluation.classifier_metrics: [lo, hi), last bin closed
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    bins: list[dict[str, Any]] = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": 0, "mean_pred": None, "pos_rate": None})
            continue
        conf, acc = float(p[mask].mean()), float(y[mask].mean())
        ece += n / len(p) * abs(conf - acc)
        bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": n, "mean_pred": conf, "pos_rate": acc})
    return float(ece), bins


def at_threshold(y: np.ndarray, p: np.ndarray, t: float) -> dict[str, float]:
    pred = p >= t
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": t,
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "accuracy": (tp + tn) / len(y) if len(y) else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def evaluate(
    y: np.ndarray, p: np.ndarray, threshold: float, groups: Sequence[str | None] | None = None
) -> dict[str, Any]:
    """Full metric block at a FIXED threshold (chosen elsewhere on validation).

    ``groups`` (e.g. negative_type, None for positives) adds per-negative-type rejection rates.
    """
    from medquad_qa.evaluation.classifier_metrics import expected_calibration_error

    _, bins = reliability(y, p)
    ece = expected_calibration_error(y.tolist(), p.tolist(), N_BINS)  # evaluator's implementation (shared numbers)
    out: dict[str, Any] = {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece_10bin": ece,
        "reliability_bins": bins,
        "at_threshold": at_threshold(y, p, threshold),
    }
    if groups is not None:
        g = np.asarray([x or "positive" for x in groups])
        per: dict[str, Any] = {}
        for name in sorted(set(g)):
            m = g == name
            rate = float((p[m] >= threshold).mean())
            per[name] = {"n": int(m.sum()), "predicted_answerable_rate": rate, "mean_score": float(p[m].mean())}
        out["by_type"] = per
    return out


def plot_reliability(curves: dict[str, list[dict[str, Any]]], out: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ":", color="grey", label="perfect calibration")
    for name, bins in curves.items():
        pts = [(b["mean_pred"], b["pos_rate"]) for b in bins if b["n"]]
        ax.plot(*zip(*pts, strict=True), "o-", label=name)
    ax.set_xlabel("mean predicted answerability (bin)")
    ax.set_ylabel("observed positive rate")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
