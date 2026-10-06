"""Unit tests for normalization, IDs, question types and quality rules (SYNTHETIC strings only)."""

import re

import pytest

from medquad_qa.data.config import ExclusionConfig, QualityConfig
from medquad_qa.data.ids import content_hash, group_id, record_id
from medquad_qa.data.normalize import (
    bracket_stripped_topic_key,
    match_key,
    normalize_question,
    normalize_text,
    topic_key,
)
from medquad_qa.data.quality import has_repeated_bullets, is_malformed_question, non_informative_kind, record_flags
from medquad_qa.data.question_types import load_rules


def test_normalize_preserves_meaning() -> None:
    raw = "Do NOT exceed 2.5 mg/kg  (≤ 10 mg)\tdaily;   e.g. Na+ < 135 mmol/L.\r\n\r\n\r\n- item one  "
    out = normalize_text(raw)
    assert out == "Do NOT exceed 2.5 mg/kg (≤ 10 mg) daily; e.g. Na+ < 135 mmol/L.\n\n- item one"
    for token in ["NOT", "2.5", "mg/kg", "≤", "10", "Na+", "<", "135"]:
        assert token in out


def test_normalize_nfc_and_no_lowercase() -> None:
    decomposed = "Café Syndrome"
    assert normalize_text(decomposed) == "Café Syndrome"
    assert normalize_text("BRCA1 Mutation") == "BRCA1 Mutation"


def test_normalize_question_space_qmark_only() -> None:
    assert normalize_question("What is (are) Glaucoma ?") == "What is (are) Glaucoma?"
    assert normalize_question("Who is at risk for X? ?") == "Who is at risk for X??"
    assert normalize_question("how vaccines prevent disease") == "how vaccines prevent disease"


def test_keys() -> None:
    assert topic_key("  Alphaitis ") == topic_key("ALPHAITIS") == "alphaitis"
    assert topic_key("") is None and topic_key(None) is None
    assert match_key("What  is X?") == match_key("what is x?")
    assert bracket_stripped_topic_key("Paget disease (bone)") == "paget disease"


@pytest.mark.parametrize(
    ("a", "b"),
    [  # F-001 regression: punctuation/diacritic variants of one topic must fold together
        ("beta-ketothiolase deficiency", "Beta ketothiolase deficiency"),
        ("Coffin-Lowry syndrome", "Coffin Lowry Syndrome"),
        ("Graves' Disease", "Graves disease"),
        ("MELAS, mitochondrial", "melas mitochondrial"),
        ("Beh\u00e7et disease", "Behcet disease"),
    ],
)
def test_topic_key_folds_punctuation(a: str, b: str) -> None:
    assert topic_key(a) == topic_key(b)
    assert match_key(f"What is {a}?") == match_key(f"what is {b} ?")


def test_record_id_rules() -> None:
    rid = record_id("SRC", "Topic", "Q?", "A.")
    assert re.fullmatch(r"mq-[0-9a-f]{16}", rid)
    assert rid == record_id("SRC", "Topic", "Q?", "A.")
    assert rid != record_id("SRC", "Topic", "Q?", "A. ")  # raw text, not normalized
    assert rid != record_id("SRC2", "Topic", "Q?", "A.")
    assert content_hash("q", "a") != content_hash("q\x1fa", "")
    assert group_id("sg", ["b", "a"]) == group_id("sg", ["a", "b"])


def test_question_types() -> None:
    rules = load_rules()
    assert rules.rule_version
    assert rules.classify("What is (are) Glaucoma?") == "information"
    assert rules.classify("Is X inherited?") == "inheritance"
    assert rules.classify("what research (or clinical trials) is being done for X?") == "research"
    assert rules.classify("What are the symptoms for X?") == "symptoms"
    assert rules.classify("Who is at risk for X??") == "susceptibility"
    assert rules.classify("how vaccines prevent disease") == rules.default_type


@pytest.mark.parametrize(
    ("answer", "kind"),
    [
        ("Is Epsilonoma inherited?", "question_only"),
        ("Topics", "heading_only"),
        ("General guidelines for safe seafood consumption:", "heading_only"),
        ("Frequently Asked Questions (FAQs)\n\nFact Sheets", "heading_only"),
        ("Zetaosis is rare.", None),  # short but valid
        ("What causes X? X is caused by a SYNTHETIC change.", None),
        ("Profound deficiency occurs in approximately 1 in 60,000 newborns", None),  # > heading word limit
    ],
)
def test_non_informative(answer: str, kind: str | None) -> None:
    assert non_informative_kind(answer, ExclusionConfig()) == kind


def test_repeated_bullets() -> None:
    rep = (
        "They include - adults over age 40 - everyone over age 60 - people with history. adults over age 40 "
        "everyone over age 60 people with history."
    )
    assert has_repeated_bullets(rep, 15)
    assert not has_repeated_bullets("They include - adults over age 40 - everyone over age 60.", 15)


def test_flags() -> None:
    assert is_malformed_question("Do you have information about X")
    assert is_malformed_question("Who is at risk for X??")
    assert not is_malformed_question("What causes X?")
    flags = record_flags("What causes X", "What causes X? Short.", None, 2, QualityConfig())
    assert set(flags) == {
        "malformed_question",
        "answer_starts_with_question",
        "short_answer",
        "missing_topic",
        "collapsed_exact_duplicates",
    }
