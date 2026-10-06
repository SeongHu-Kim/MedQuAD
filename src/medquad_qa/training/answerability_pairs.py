"""Synthetic answerability pairs (pair_rule_version ``ans-pairs-v1``), built within one split.

Rules (agreed with evaluation-safety-engineer; label_provenance = synthetic_rule):
- question = a record's normalized question; evidence = first <=200-word chunk of another (or the same) record's
  normalized answer.
- positives: the record's own answer (``pos-self``) and, if one exists, one other record with the same topic AND
  question_type (``pos-sib``). Records flagged boilerplate/non-informative are never used as evidence.
- negatives, as many as positives per question, types assigned round-robin from a seeded RNG:
  easy_random (different topic), same_topic_diff_qtype (same topic, different question_type),
  lexical_hard_bm25 (top BM25 answer in the same split with a different topic and duplicate group).
"Same topic + same question type" is a proxy for "the evidence suffices"; it is not a clinical judgement.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from medquad_qa.contracts.answerability import AnswerabilityPair, NegativeType
from medquad_qa.contracts.records import MedicalRecord
from medquad_qa.training.sft_data import EXCLUDED_FLAGS

PAIR_RULE_VERSION = "ans-pairs-v1"
EVIDENCE_MAX_WORDS = 200
NEGATIVE_TYPES: tuple[NegativeType, ...] = ("easy_random", "same_topic_diff_qtype", "lexical_hard_bm25")
SplitName = Literal["train", "validation", "test"]


def first_chunk(text: str, max_words: int = EVIDENCE_MAX_WORDS) -> str:
    words = text.split()
    return " ".join(words[:max_words])


def _key(r: MedicalRecord) -> str:
    return (r.topic or "").casefold()


class _LexicalIndex:
    """BM25 over the split's evidence pool (bm25s, English stopwords, no stemming)."""

    def __init__(self, pool: Sequence[MedicalRecord]) -> None:
        import bm25s

        self.pool = list(pool)
        self.bm25s = bm25s
        self.model = bm25s.BM25(k1=1.5, b=0.75)
        self.model.index(
            bm25s.tokenize([first_chunk(r.answer) for r in self.pool], show_progress=False), show_progress=False
        )

    def hard_negative(self, q: MedicalRecord, k: int = 50) -> MedicalRecord | None:
        tokens = self.bm25s.tokenize([q.question], show_progress=False)
        docs, _ = self.model.retrieve(tokens, k=min(k, len(self.pool)), show_progress=False)
        for idx in docs[0]:
            cand = self.pool[int(idx)]
            if _key(cand) != _key(q) and cand.duplicate_group_id != q.duplicate_group_id:
                return cand
        return None


def build_pairs(records: Sequence[MedicalRecord], split: SplitName, seed: int = 20261006) -> list[AnswerabilityPair]:
    rng = random.Random(f"{seed}:{split}")  # noqa: S311 - reproducible sampling, not security
    recs = sorted(records, key=lambda r: r.record_id)
    pool = [r for r in recs if not EXCLUDED_FLAGS.intersection(r.quality_flags) and r.answer.strip()]
    by_topic: dict[str, list[MedicalRecord]] = defaultdict(list)
    for r in pool:
        by_topic[_key(r)].append(r)
    index = _LexicalIndex(pool)
    pairs: list[AnswerabilityPair] = []

    def add(q: MedicalRecord, e: MedicalRecord, label: bool, tag: str, neg: NegativeType | None) -> None:
        pairs.append(
            AnswerabilityPair(
                pair_id=f"{PAIR_RULE_VERSION}:{split}:{q.record_id}:{tag}:{e.record_id}",
                question=q.question,
                question_record_id=q.record_id,
                evidence_record_id=e.record_id,
                evidence_text=first_chunk(e.answer),
                label=label,
                negative_type=neg,
                pair_rule_version=PAIR_RULE_VERSION,
                split=split,
                split_group_id=q.split_group_id,
            )
        )

    for q in pool:
        n_pos = 1
        add(q, q, True, "pos-self", None)
        sibs = [
            r
            for r in by_topic[_key(q)]
            if r.record_id != q.record_id and q.question_type is not None and r.question_type == q.question_type
        ]
        if sibs:
            add(q, rng.choice(sibs), True, "pos-sib", None)
            n_pos += 1
        for _ in range(n_pos):
            for neg_type in rng.sample(NEGATIVE_TYPES, k=len(NEGATIVE_TYPES)):  # first type that yields a candidate
                e = _negative(q, neg_type, pool, by_topic, index, rng)
                if e is not None:
                    add(q, e, False, f"neg-{neg_type}", neg_type)
                    break
    return pairs


def _negative(
    q: MedicalRecord,
    neg_type: NegativeType,
    pool: list[MedicalRecord],
    by_topic: dict[str, list[MedicalRecord]],
    index: _LexicalIndex,
    rng: random.Random,
) -> MedicalRecord | None:
    if neg_type == "easy_random":
        for _ in range(20):
            cand = rng.choice(pool)
            if _key(cand) != _key(q):
                return cand
        return None
    if neg_type == "same_topic_diff_qtype":
        cands = [r for r in by_topic[_key(q)] if r.question_type != q.question_type and r.record_id != q.record_id]
        return rng.choice(cands) if cands and q.question_type is not None else None
    return index.hard_negative(q)


def write_pairs(pairs: list[AnswerabilityPair], text_path: Path, manifest_path: Path) -> None:
    """Full pairs (with dataset text; gitignored) + an IDs-only manifest for leakage checks (tracked)."""
    text_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with text_path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(p.model_dump_json() + "\n")
    keep = ("pair_id", "question_record_id", "evidence_record_id", "split", "split_group_id", "label", "negative_type")
    with manifest_path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps({k: getattr(p, k) for k in keep}) + "\n")


def read_pairs(path: Path) -> list[AnswerabilityPair]:
    with path.open(encoding="utf-8") as f:
        return [AnswerabilityPair.model_validate_json(line) for line in f if line.strip()]
