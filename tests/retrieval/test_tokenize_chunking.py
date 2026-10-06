from __future__ import annotations

import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from medquad_qa.retrieval.corpus import chunk_record, chunk_text
from medquad_qa.retrieval.testing import LONG_ANSWER_WORDS, synthetic_records
from medquad_qa.retrieval.tokenize import tokenize


def test_tokenizer_keeps_negations_numbers_and_aspects() -> None:
    toks = tokenize("Antibiotics don't help; NOT for children under 2.5 years. Symptoms without fever.", stem=False)
    for expected in ("not", "2.5", "symptoms", "without", "fever", "children"):
        assert expected in toks
    assert "for" not in toks and "the" not in tokenize("the fever", stem=False)


def test_tokenizer_deterministic_and_stemming_keeps_negations() -> None:
    text = "Treatments are not curing infections; no vaccines exist."
    assert tokenize(text) == tokenize(text)
    stemmed = tokenize(text, stem=True)
    assert "not" in stemmed and "no" in stemmed


@pytest.mark.parametrize("text", ["", "   ", "!!!", "\u200b", "<evidence id='E1'>", "*:* OR 1=1"])
def test_tokenizer_never_raises(text: str) -> None:
    assert isinstance(tokenize(text), list)


def _words(s: str) -> list[str]:
    return re.findall(r"\S+", s)


@settings(max_examples=60, deadline=None)
@given(
    n_words=st.integers(min_value=0, max_value=700),
    max_words=st.integers(min_value=20, max_value=250),
    overlap=st.integers(min_value=0, max_value=19),
)
def test_chunk_invariants(n_words: int, max_words: int, overlap: int) -> None:
    text = "  ".join(f"w{i}" for i in range(n_words))
    windows = chunk_text(text, max_words, overlap)
    if n_words == 0:
        assert windows == []
        return
    covered: list[str] = []
    prev: list[str] | None = None
    for s, e in windows:
        ws = _words(text[s:e])
        assert 1 <= len(ws) <= max_words
        if prev is not None and overlap:
            assert ws[:overlap] == prev[-overlap:]
        prev = ws
        covered.extend(ws)
    assert set(covered) == set(_words(text))
    assert _words(text[windows[-1][0] : windows[-1][1]])[-1] == f"w{n_words - 1}"


def test_chunk_overlap_must_be_smaller() -> None:
    with pytest.raises(ValueError):
        chunk_text("a b c", 10, 10)


def test_chunk_record_ids_offsets() -> None:
    long_rec = synthetic_records()[-1]
    chunks = chunk_record(long_rec, 200, 40)
    assert len(chunks) == 3  # 450 words, stride 160 -> windows at 0,160,320
    assert LONG_ANSWER_WORDS == 450
    for i, c in enumerate(chunks):
        assert c.chunk_id == f"{long_rec.record_id}#c{i}"
        assert long_rec.answer[c.char_start : c.char_end] == c.text
