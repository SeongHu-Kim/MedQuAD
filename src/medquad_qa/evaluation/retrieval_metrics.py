"""Retrieval metrics over ranked record IDs.

Definitions (fixed for the report; see docs/evaluation/metrics.md):
- ``hit_at_k`` ("Recall@k" in the report): 1 if ANY gold record is in the top k, else 0. EvaluationExample
  treats ``gold_record_ids`` as a set where retrieving any member counts as relevant.
- ``gold_recall_at_k``: |gold ∩ top-k| / |gold| (secondary; penalises large duplicate-heavy gold sets).
- ``reciprocal_rank``: 1/rank of the first gold record within the cutoff, else 0. MRR is its mean.
- ``ndcg_at_k``: only defined when graded relevance labels exist; raises otherwise.
Queries with an empty gold set are not scorable and must be excluded by the caller (``score_queries`` raises).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict

DEFAULT_KS: tuple[int, ...] = (1, 5, 10, 20)


def hit_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    if k < 1:
        raise ValueError("k must be >= 1")
    return 1.0 if any(r in gold for r in ranked[:k]) else 0.0


def gold_recall_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    if not gold:
        raise ValueError("empty gold set")
    return len(gold.intersection(ranked[:k])) / len(gold)


def reciprocal_rank(ranked: Sequence[str], gold: set[str], cutoff: int | None = None) -> float:
    for i, r in enumerate(ranked[:cutoff] if cutoff else ranked, 1):
        if r in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], graded: Mapping[str, int], k: int) -> float:
    """Exponential-gain nDCG@k (gain 2^rel - 1). ``graded`` must be non-empty with at least one rel > 0."""
    if not graded or max(graded.values()) <= 0:
        raise ValueError("nDCG requires graded relevance labels with at least one positive grade")
    dcg = sum((2 ** graded.get(r, 0) - 1) / math.log2(i + 1) for i, r in enumerate(ranked[:k], 1))
    ideal = sorted(graded.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, 1))
    return dcg / idcg


class QueryRetrievalScores(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    example_id: str
    hit_at: dict[int, float]
    gold_recall_at: dict[int, float]
    reciprocal_rank: float
    first_relevant_rank: int | None
    ndcg_at: dict[int, float] | None = None


def score_query(
    example_id: str,
    ranked: Sequence[str],
    gold: set[str],
    ks: Sequence[int] = DEFAULT_KS,
    graded: Mapping[str, int] | None = None,
) -> QueryRetrievalScores:
    if not gold:
        raise ValueError(f"{example_id}: empty gold set is not scorable for retrieval")
    rr = reciprocal_rank(ranked, gold, cutoff=max(ks))
    return QueryRetrievalScores(
        example_id=example_id,
        hit_at={k: hit_at_k(ranked, gold, k) for k in ks},
        gold_recall_at={k: gold_recall_at_k(ranked, gold, k) for k in ks},
        reciprocal_rank=rr,
        first_relevant_rank=round(1 / rr) if rr > 0 else None,
        ndcg_at={k: ndcg_at_k(ranked, graded, k) for k in ks} if graded else None,
    )


def aggregate(scores: Sequence[QueryRetrievalScores]) -> dict[str, float | int]:
    """Macro-average over queries. Keys: n, recall@k, gold_recall@k, mrr@<maxk>, [ndcg@k]."""
    if not scores:
        raise ValueError("no scored queries")
    n = len(scores)
    ks = sorted(scores[0].hit_at)
    out: dict[str, float | int] = {"n": n}
    for k in ks:
        out[f"recall@{k}"] = sum(s.hit_at[k] for s in scores) / n
        out[f"gold_recall@{k}"] = sum(s.gold_recall_at[k] for s in scores) / n
    out[f"mrr@{max(ks)}"] = sum(s.reciprocal_rank for s in scores) / n
    if all(s.ndcg_at is not None for s in scores):
        for k in ks:
            out[f"ndcg@{k}"] = sum(s.ndcg_at[k] for s in scores if s.ndcg_at is not None) / n
    return out
