"""Exclusion rules and record-level quality flags (bare names, see docs/data/data_card.md).

Flags never change text; they only describe it. ``boilerplate_answer`` is assigned in grouping.
"""

from __future__ import annotations

import re

from medquad_qa.data.config import ExclusionConfig, QualityConfig
from medquad_qa.data.normalize import word_count

EXCLUDE_EMPTY = "empty_answer"
EXCLUDE_DUPLICATE = "exact_duplicate_row"
EXCLUDE_NON_INFORMATIVE = "non_informative_answer"
EXCLUSION_REASONS = (EXCLUDE_EMPTY, EXCLUDE_DUPLICATE, EXCLUDE_NON_INFORMATIVE)

FLAG_BOILERPLATE = "boilerplate_answer"
FLAG_MALFORMED_Q = "malformed_question"
FLAG_REPEATED_BULLETS = "repeated_bullets"
FLAG_STARTS_WITH_Q = "answer_starts_with_question"
FLAG_SHORT = "short_answer"
FLAG_MISSING_TOPIC = "missing_topic"
FLAG_COLLAPSED = "collapsed_exact_duplicates"
ALL_FLAGS = (
    FLAG_BOILERPLATE,
    FLAG_MALFORMED_Q,
    FLAG_REPEATED_BULLETS,
    FLAG_STARTS_WITH_Q,
    FLAG_SHORT,
    FLAG_MISSING_TOPIC,
    FLAG_COLLAPSED,
)

_ONLY_A_QUESTION = re.compile(r"^[^.!?\n]+\?$")
_LEADING_QUESTION = re.compile(r"^[^.!?\n]{3,}\?\s+\S")
_BULLET_SPLIT = re.compile(r"(?:^|\s)-\s+", re.MULTILINE)


def non_informative_kind(answer: str, cfg: ExclusionConfig) -> str | None:
    """Return why a non-empty normalized answer carries no answer content, else None.

    * ``question_only``: the whole answer is one question sentence (a heading copied as the answer).
    * ``heading_only``: at most ``heading_max_words`` words and no line ends a sentence with
      ``.``/``!``/``?`` (e.g. "Topics", or "Frequently Asked Questions (FAQs)" + a second heading line).
    """
    if _ONLY_A_QUESTION.match(answer):
        return "question_only"
    lines = [ln for ln in answer.split("\n") if ln]
    if word_count(answer) <= cfg.heading_max_words and not any(ln.endswith((".", "!", "?")) for ln in lines):
        return "heading_only"
    return None


def has_repeated_bullets(answer: str, min_chars: int) -> bool:
    """Heuristic: at least two bullet items whose text occurs again elsewhere in the answer.

    Catches the export artefact where a "- a - b - c" list is followed by "a b c" verbatim.
    """
    items = [p.strip() for p in _BULLET_SPLIT.split(answer)[1:]]
    repeated = 0
    for item in items:
        if len(item) >= min_chars and answer.count(item) >= 2:
            repeated += 1
            if repeated >= 2:
                return True
    return False


def is_malformed_question(question: str) -> bool:
    return not question.endswith("?") or "??" in question


def record_flags(question: str, answer: str, topic: str | None, n_collapsed: int, cfg: QualityConfig) -> list[str]:
    flags: list[str] = []
    if is_malformed_question(question):
        flags.append(FLAG_MALFORMED_Q)
    if has_repeated_bullets(answer, cfg.repeated_bullet_min_chars):
        flags.append(FLAG_REPEATED_BULLETS)
    if _LEADING_QUESTION.match(answer):
        flags.append(FLAG_STARTS_WITH_Q)
    if word_count(answer) <= cfg.short_answer_max_words:
        flags.append(FLAG_SHORT)
    if topic is None:
        flags.append(FLAG_MISSING_TOPIC)
    if n_collapsed:
        flags.append(FLAG_COLLAPSED)
    return flags
