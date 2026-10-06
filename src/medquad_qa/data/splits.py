"""Deterministic group-level train/validation/test assignment, stratified by dominant source."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Sequence

SPLITS = ("train", "validation", "test")


def dominant_source(sources: Sequence[str]) -> str:
    counts = Counter(sources)
    return min(counts, key=lambda s: (-counts[s], s))


def assign_splits(
    group_ids: Sequence[str], sources: Sequence[str], ratios: dict[str, float], seed: int
) -> tuple[dict[str, str], dict[str, str]]:
    """Return ({split_group_id: split}, {split_group_id: dominant_source}).

    Per stratum (dominant source), groups are sorted by ID, shuffled with a stratum-specific seed and
    assigned greedily to the split furthest below its row target. Every record of a group therefore
    lands in exactly one split.
    """
    if set(ratios) != set(SPLITS) or abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError(f"ratios must cover {SPLITS} and sum to 1: {ratios}")
    members: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(group_ids):
        members[g].append(i)
    dom = {g: dominant_source([sources[i] for i in idx]) for g, idx in members.items()}

    strata: dict[str, list[str]] = defaultdict(list)
    for g in sorted(members):
        strata[dom[g]].append(g)

    assignment: dict[str, str] = {}
    for stratum in sorted(strata):
        groups = strata[stratum]
        random.Random(f"{seed}|{stratum}").shuffle(groups)  # noqa: S311 - reproducible split, not crypto
        total = sum(len(members[g]) for g in groups)
        target = {s: ratios[s] * total for s in SPLITS}
        filled = dict.fromkeys(SPLITS, 0)
        for g in groups:
            best = max(SPLITS, key=lambda s: ((target[s] - filled[s]) / target[s] if target[s] else float("-inf")))
            assignment[g] = best
            filled[best] += len(members[g])
    return assignment, dom
