"""Evidence budget (D-022): fit evidence to the generator's input-token budget in rank order,
dropping whole lower-ranked hits (never truncating inside a chunk)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from medquad_qa.contracts import ChatMessage, RetrievalHit

EVIDENCE_TRUNCATED = "evidence_truncated"
TokenCounter = Callable[[list[ChatMessage]], int]

# Conservative fallback: ~3 characters per token for English medical text, plus template overhead.
_CHARS_PER_TOKEN = 3.0
_PER_MESSAGE_OVERHEAD = 8
_TEMPLATE_OVERHEAD = 16


def estimate_tokens(messages: list[ChatMessage]) -> int:
    return _TEMPLATE_OVERHEAD + sum(
        _PER_MESSAGE_OVERHEAD + math.ceil(len(m.content) / _CHARS_PER_TOKEN) for m in messages
    )


def token_counter_for(generator: Any) -> tuple[TokenCounter, str]:
    """Use ``generator.count_tokens(messages)`` when present (exact), else the estimate."""
    fn = getattr(generator, "count_tokens", None)
    if callable(fn):
        return fn, "generator"
    return estimate_tokens, "estimate"


def fit_evidence(
    hits: Sequence[RetrievalHit],
    build: Callable[[Sequence[RetrievalHit]], list[ChatMessage]],
    count: TokenCounter,
    max_input_tokens: int,
) -> tuple[list[RetrievalHit], list[ChatMessage], bool]:
    """Return (supplied_hits, messages, truncated). Keeps the longest rank-order prefix that fits.

    If not even the first hit fits, returns no hits (the caller abstains).
    """
    lo, hi = 0, len(hits)
    msgs = build(hits)
    if count(msgs) <= max_input_tokens:
        return list(hits), msgs, False
    # token count is monotone in the prefix length: binary search the longest fitting prefix
    best_n, best_msgs = 0, build([])
    while lo < hi:
        mid = (lo + hi + 1) // 2
        m = build(hits[:mid])
        if count(m) <= max_input_tokens:
            best_n, best_msgs, lo = mid, m, mid
        else:
            hi = mid - 1
    return list(hits[:best_n]), best_msgs, True
