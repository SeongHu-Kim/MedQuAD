"""Deterministic lexical tokenizer for BM25.

Lowercases (index side only; stored text is never altered), keeps numbers (incl. decimals such as
``2.5``), keeps negations (``not``, ``no``, ``without``, ``n't`` -> ``not``) and medical aspect words
(symptoms, treatment, causes, ...). Only a small closed list of function words is dropped.
Stemming uses PyStemmer's English Snowball stemmer when available and enabled.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

TOKENIZER_VERSION = "tok-v1"

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_NT_RE = re.compile(r"n['’]t\b")

NEGATIONS: frozenset[str] = frozenset({"no", "not", "nor", "never", "none", "without", "cannot", "neither"})

# Closed list; deliberately excludes negations, quantifiers and question/aspect words.
STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "for",
        "from",
        "with",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "it's",
        "into",
        "than",
        "then",
        "there",
        "their",
        "them",
        "they",
        "he",
        "she",
        "his",
        "her",
        "we",
        "our",
        "you",
        "your",
        "i",
        "me",
        "my",
        "do",
        "does",
        "did",
        "doing",
        "have",
        "has",
        "had",
        "having",
        "will",
        "would",
        "shall",
        "should",
        "can",
        "could",
        "may",
        "might",
        "must",
        "also",
        "such",
        "so",
        "very",
        "just",
        "about",
        "which",
        "who",
        "whom",
        "whose",
        "s",
    }
)


@lru_cache(maxsize=2)
def _stemmer(enabled: bool) -> object | None:
    if not enabled:
        return None
    try:
        import Stemmer  # PyStemmer
    except ImportError:  # optional dependency
        return None
    return Stemmer.Stemmer("english")


def stemmer_available() -> bool:
    return _stemmer(True) is not None


def tokenize(text: str, *, stem: bool = True) -> list[str]:
    """Tokenize ``text`` deterministically. Never raises on any str input."""
    norm = unicodedata.normalize("NFKC", text).lower()
    norm = _NT_RE.sub(" not", norm)
    tokens = [t for t in _TOKEN_RE.findall(norm) if t not in STOPWORDS]
    stemmer = _stemmer(stem)
    if stemmer is None:
        return tokens
    # Negations and numbers are left unstemmed so they stay exact.
    return [t if (t in NEGATIONS or t[0].isdigit()) else stemmer.stemWord(t) for t in tokens]  # type: ignore[attr-defined]


def tokenizer_id(stem: bool) -> str:
    """Identifier recorded in manifests; differs when stemming is effectively off."""
    return f"{TOKENIZER_VERSION}+{'snowball-en' if stem and stemmer_available() else 'nostem'}"
