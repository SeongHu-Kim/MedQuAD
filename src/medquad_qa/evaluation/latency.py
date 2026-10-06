"""Latency summaries. Percentiles use numpy's default linear interpolation (documented in metrics.md)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def summarize(latencies_ms: Sequence[float]) -> dict[str, float | int]:
    a = np.asarray(latencies_ms, dtype=float)
    if a.size == 0:
        raise ValueError("no latency samples")
    if (a < 0).any() or np.isnan(a).any():
        raise ValueError("latencies must be non-negative numbers")
    return {
        "n": int(a.size),
        "median_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "mean_ms": float(a.mean()),
        "max_ms": float(a.max()),
    }


def summarize_components(components: Sequence[Mapping[str, float]]) -> dict[str, dict[str, float | int]]:
    """Per-key summary over QAResponse.component_latency_ms dicts; keys absent from a response are skipped."""
    keys = sorted({k for c in components for k in c})
    return {k: summarize([c[k] for c in components if k in c]) for k in keys}
