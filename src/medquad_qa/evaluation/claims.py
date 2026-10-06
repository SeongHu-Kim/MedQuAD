"""Claim segmentation for citation and unsupported-claim metrics.

A *claim* is one sentence of the answer with at least ``MIN_CLAIM_WORDS`` words after removing inline citation
markers. This is a deterministic, documented approximation (no atomic-fact decomposition).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MIN_CLAIM_WORDS = 4
CITATION_MARKER = re.compile(r"\[(mq-[0-9a-f]{16})\]")
# Unmapped per-request labels (should already be rewritten/stripped by the pipeline; counted if they leak through).
LABEL_MARKER = re.compile(r"\[E\d+\]")
_ABBREV = re.compile(r"\b(?:e\.g|i\.e|etc|vs|Dr|Mr|Mrs|Ms|St|approx|No|Fig)\.", re.IGNORECASE)
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])|\n+")
_WORD = re.compile(r"\w+")
_SENTINEL = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class Claim:
    text: str  # sentence with citation markers removed
    cited_ids: tuple[str, ...]  # record IDs cited inside this sentence, in order, de-duplicated


def split_sentences(text: str) -> list[str]:
    protected = _ABBREV.sub(lambda m: m.group(0).replace(".", "\x00"), text)
    parts = (p.replace("\x00", ".").strip() for p in _SPLIT.split(protected))
    return [p for p in parts if p]


def strip_markers(text: str) -> str:
    out = LABEL_MARKER.sub("", CITATION_MARKER.sub("", text))
    return re.sub(r"\s+([.,;:!?])", r"\1", re.sub(r"[ \t]{2,}", " ", out)).strip()


def extract_claims(answer: str, min_words: int = MIN_CLAIM_WORDS) -> list[Claim]:
    """Claims in order. A trailing citation that follows the sentence's full stop is attached to that sentence."""
    # Move "... end. [mq-x]" markers before the full stop so they stay with their sentence after splitting.
    normalized = re.sub(r"([.!?])((?:\s*\[(?:mq-[0-9a-f]{16}|E\d+)\])+)", r"\2\1", answer)
    claims: list[Claim] = []
    for sent in split_sentences(normalized):
        if _SENTINEL in sent:
            continue
        clean = strip_markers(sent)
        if len(_WORD.findall(clean)) < min_words:
            continue
        ids = tuple(dict.fromkeys(CITATION_MARKER.findall(sent)))
        claims.append(Claim(text=clean, cited_ids=ids))
    return claims
