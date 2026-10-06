"""Supplementary lexical overlap metrics (never primary evidence of correctness)."""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from typing import Any

from medquad_qa.evaluation.claims import strip_markers

_TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(strip_markers(text).lower())


def token_f1(prediction: str, reference: str) -> float:
    p, r = tokens(prediction), tokens(reference)
    if not p or not r:
        return float(p == r)
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


@lru_cache(maxsize=1)
def _rouge() -> Any:
    from rouge_score import rouge_scorer

    return rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


def rouge_l_f(prediction: str, reference: str) -> float:
    return float(_rouge().score(strip_markers(reference), strip_markers(prediction))["rougeL"].fmeasure)
