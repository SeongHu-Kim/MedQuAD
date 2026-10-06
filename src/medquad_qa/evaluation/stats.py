"""Paired significance tools for mode-vs-mode comparisons on the same queries.

- ``paired_cluster_bootstrap``: resamples clusters (split_group_id) with replacement; within a resampled cluster
  all of its items are kept, preserving the pairing between system A and system B. Reports the mean difference
  (B - A), a percentile CI, and a two-sided bootstrap p-value (proportion of resampled differences on the other
  side of 0, doubled, floored at 1/(R+1)).
- ``exact_mcnemar``: exact two-sided binomial test on discordant pairs for binary per-item outcomes.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from scipy.stats import binomtest


def paired_cluster_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    clusters: Sequence[str],
    *,
    n_resamples: int = 10_000,
    seed: int = 20261006,
    ci: float = 0.95,
) -> dict[str, float | int]:
    if not (len(a) == len(b) == len(clusters)) or not a:
        raise ValueError("a, b and clusters must be equal-length and non-empty")
    diff = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    if np.isnan(diff).any():
        raise ValueError("NaN in paired values; drop unscorable items for both systems first")
    members: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(clusters):
        members[c].append(i)
    keys = list(members)
    sums = np.array([diff[members[k]].sum() for k in keys])
    counts = np.array([len(members[k]) for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_resamples, len(keys)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    observed = float(diff.mean())
    tail = min((boot <= 0).mean(), (boot >= 0).mean())
    p = max(min(1.0, 2 * tail), 1 / (n_resamples + 1))
    return {
        "n_items": len(diff),
        "n_clusters": len(keys),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "diff_b_minus_a": observed,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci": ci,
        "p_value": float(p),
        "n_resamples": n_resamples,
        "seed": seed,
    }


def exact_mcnemar(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> dict[str, float | int]:
    if len(a_correct) != len(b_correct) or not a_correct:
        raise ValueError("paired outcomes must be equal-length and non-empty")
    only_a = sum(1 for x, y in zip(a_correct, b_correct, strict=True) if x and not y)
    only_b = sum(1 for x, y in zip(a_correct, b_correct, strict=True) if y and not x)
    n = only_a + only_b
    p = 1.0 if n == 0 else float(binomtest(only_a, n, 0.5, alternative="two-sided").pvalue)
    return {"n": len(a_correct), "a_only_correct": only_a, "b_only_correct": only_b, "p_value": p}
