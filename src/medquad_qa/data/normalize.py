"""Text normalization that preserves medically meaningful content.

Allowed edits only: Unicode NFC, newline unification, per-line whitespace collapse, blank-line
collapse, and (questions only) ``" ?" -> "?"``. No lowercasing, and no punctuation, number,
negation or spelling edits. Case-folded keys are used for *matching* only, never stored as text.
"""

from __future__ import annotations

import re
import unicodedata

_INLINE_WS = re.compile(r"[^\S\n]+")  # whitespace except newline (includes NBSP, tabs)
_BLANK_LINES = re.compile(r"\n{2,}")
_SPACE_QMARK = re.compile(r" \?")
_PARENS = re.compile(r"\([^()]*\)")
_WORD = re.compile(r"\w+")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [_INLINE_WS.sub(" ", line).strip() for line in text.split("\n")]
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def normalize_question(text: str) -> str:
    return _SPACE_QMARK.sub("?", normalize_text(text))


def normalize_topic(text: str) -> str | None:
    t = normalize_text(text)
    return t or None


def match_key(text: str) -> str:
    """Case- and whitespace-insensitive key for equality checks (matching only)."""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def topic_key(topic: str | None) -> str | None:
    """Folded topic key used to merge the same focus area across sources."""
    if topic is None:
        return None
    key = match_key(topic).strip(" .,;:")
    return key or None


def bracket_stripped_topic_key(topic: str | None) -> str | None:
    """Topic key with parenthetical text removed. Diagnostic only; never used for grouping."""
    if topic is None:
        return None
    return topic_key(_PARENS.sub(" ", topic))


def word_count(text: str) -> int:
    return len(_WORD.findall(text))
