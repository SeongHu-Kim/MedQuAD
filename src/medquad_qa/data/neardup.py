"""Near-duplicate answer detection with topic-masked word shingles.

Each answer is tokenized (case-folded, matching only), the record's own topic tokens are replaced by
``<t>`` so templated answers about different conditions are compared on their shared wording, and
word n-gram shingles are built. Candidates are pairs that share rare shingles (document frequency
<= ``max_df``); their exact Jaccard similarity over the full shingle sets is then computed.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

_TOKEN = re.compile(r"\w+")
TOPIC_MASK = "<t>"


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.casefold())


def mask_topic(toks: list[str], topic_toks: list[str]) -> list[str]:
    n = len(topic_toks)
    if n == 0:
        return toks
    out: list[str] = []
    i = 0
    while i < len(toks):
        if toks[i : i + n] == topic_toks:
            out.append(TOPIC_MASK)
            i += n
        else:
            out.append(toks[i])
            i += 1
    return out


def shingles(toks: list[str], size: int) -> frozenset[str]:
    if len(toks) < size:
        return frozenset()
    return frozenset(" ".join(toks[i : i + size]) for i in range(len(toks) - size + 1))


def masked_shingles(answer: str, topic: str | None, size: int) -> frozenset[str]:
    return shingles(mask_topic(tokens(answer), tokens(topic or "")), size)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


@dataclass(frozen=True)
class SimilarPair:
    i: int
    j: int
    jaccard: float
    shared_rare: int


def similar_pairs(
    sets: Sequence[frozenset[str]], *, max_df: int, min_shared_rare: int, min_jaccard: float
) -> list[SimilarPair]:
    """All index pairs (i < j) sharing >= ``min_shared_rare`` rare shingles with Jaccard >= ``min_jaccard``.

    Identical shingle sets are compared too (callers decide whether those are already linked).
    Recall caveat: a pair whose overlap consists only of shingles with df > ``max_df`` is not found.
    """
    df: Counter[str] = Counter()
    for s in sets:
        df.update(s)
    postings: dict[str, list[int]] = defaultdict(list)
    for idx, s in enumerate(sets):
        for sh in s:
            if df[sh] <= max_df:
                postings[sh].append(idx)
    shared: Counter[tuple[int, int]] = Counter()
    for ids in postings.values():
        if len(ids) > 1:
            shared.update(combinations(ids, 2))
    out: list[SimilarPair] = []
    for (i, j), n in shared.items():
        if n < min_shared_rare:
            continue
        jac = jaccard(sets[i], sets[j])
        if jac >= min_jaccard:
            out.append(SimilarPair(i, j, jac, n))
    out.sort(key=lambda p: (p.i, p.j))
    return out
