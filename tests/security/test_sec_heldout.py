"""Held-out integrity (D-035): owners' code, tests and docs must not contain TEST / train_probe eval text.

Reads the frozen eval sets and the agent-written templates at runtime (no eval strings are embedded here) and
searches owner paths for any normalized question, or any >=24-char fragment of a Track C personal-advice template.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EVALSETS = ROOT / "artifacts/evaluation/evalsets"
HELD_OUT = ("test_track_a.jsonl", "test_track_c.jsonl", "train_probe.jsonl")
OWNER_PATHS = (
    "src/medquad_qa/rag",
    "src/medquad_qa/retrieval",
    "src/medquad_qa/models",
    "src/medquad_qa/training",
    "src/medquad_qa/api",
    "tests/rag",
    "tests/retrieval",
    "tests/models",
    "tests/training",
    "tests/api",
    "configs/retrieval",
    "configs/models",
    "configs/training",
    "docs/retrieval",
    "docs/models",
)
MIN_FRAGMENT = 24


def norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[^0-9a-z]+", " ", t).strip()


def owner_text() -> str:
    parts = []
    for rel in OWNER_PATHS:
        base = ROOT / rel
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix in {".py", ".md", ".json", ".yaml", ".yml", ".txt", ".jsonl"}:
                parts.append(norm(p.read_text(encoding="utf-8", errors="ignore")))
    return " \n ".join(parts)


@pytest.fixture(scope="module")
def corpus_of_owner_text() -> str:
    if not EVALSETS.exists():
        pytest.skip("frozen eval sets not present")
    return owner_text()


def test_no_held_out_questions_in_owner_paths(corpus_of_owner_text: str) -> None:
    hits = []
    for name in HELD_OUT:
        for line in (EVALSETS / name).read_text(encoding="utf-8").splitlines():
            ex = json.loads(line)
            q = norm(ex["question"])
            if len(q) >= MIN_FRAGMENT and q in corpus_of_owner_text:
                hits.append((name, ex["example_id"]))
    assert hits == [], hits


def test_no_track_c_personal_templates_in_owner_paths(corpus_of_owner_text: str) -> None:
    agent = json.loads((ROOT / "configs/evaluation/agent_items.json").read_text(encoding="utf-8"))
    hits = []
    for tmpl in agent["personal_templates_test"]:
        for frag in tmpl.split("{t}"):
            f = norm(frag)
            if len(f) >= MIN_FRAGMENT and f in corpus_of_owner_text:
                hits.append(tmpl)
    assert hits == [], hits
