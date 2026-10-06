"""Evaluation-set leakage checks. ``check_leakage`` fails the build (raises ``LeakageError``) on any error.

Errors (hard failures):
 E1 duplicate example_id, or duplicate normalized question within the evaluation set.
 E2 split origin: every source record (derived_from_record_ids) and gold record of an item must belong to a group
    in the split that matches eval_split (test->test, dev->validation, train_probe->train); split_group_id, when
    set, must equal the source record's group.
 E3 question copy: a dev/test question exactly or normalized-equal to any indexed corpus question.
 E4 protected overlap: a dev/test item's source or gold groups intersect the groups of records used for SFT or
    classifier training; a test item's groups also must not intersect classifier-validation/threshold-fitting
    groups. (train_probe items overlap SFT by design and are reported, not failed.)
 E5 unknown record IDs (not in the split map).
Warnings (reported, not failed): near-copy questions (token-set Jaccard >= ``near_copy_jaccard``) of indexed
questions, and train_probe items that reuse an indexed question verbatim.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from medquad_qa.contracts.evaluation import EvaluationExample

SplitName = Literal["train", "validation", "test"]
EVAL_TO_SPLIT: dict[str, SplitName] = {"test": "test", "dev": "validation", "train_probe": "train"}

_NON_ALNUM = re.compile(r"[^0-9a-z]+")


class LeakageError(Exception):
    """Raised when the evaluation set fails a hard leakage check."""


def normalize_question(text: str) -> str:
    """NFKC, casefold, punctuation -> space, whitespace collapse. Used only for leakage comparison."""
    return _NON_ALNUM.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def _tokset(text: str) -> frozenset[str]:
    return frozenset(normalize_question(text).split())


class LeakageReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n_examples: int
    counts_by_eval_split: dict[str, int]
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    train_probe_sft_overlap: int = 0
    near_copy_jaccard: float

    @property
    def passed(self) -> bool:
        return not self.errors


def check_leakage(
    examples: Sequence[EvaluationExample],
    record_group: Mapping[str, str],
    group_split: Mapping[str, SplitName],
    indexed_questions: Iterable[str],
    *,
    sft_record_ids: Iterable[str] = (),
    classifier_train_record_ids: Iterable[str] = (),
    threshold_record_ids: Iterable[str] = (),
    near_copy_jaccard: float = 0.9,
    fail: bool = True,
) -> LeakageReport:
    """Run all checks. ``record_group`` maps record_id -> split_group_id; ``group_split`` maps group -> split.

    ``threshold_record_ids`` covers classifier-validation pairs and any data used to fit thresholds/heuristics.
    """
    errors: list[str] = []
    warnings: list[str] = []

    def groups_of(ids: Iterable[str]) -> set[str]:
        return {record_group[r] for r in ids if r in record_group}

    sft_g = groups_of(sft_record_ids)
    clf_g = groups_of(classifier_train_record_ids)
    thr_g = groups_of(threshold_record_ids)

    # E1
    for eid, c in Counter(e.example_id for e in examples).items():
        if c > 1:
            errors.append(f"E1 duplicate example_id {eid} x{c}")
    by_norm: dict[str, list[str]] = defaultdict(list)
    for e in examples:
        by_norm[normalize_question(e.question)].append(e.example_id)
    for ids in by_norm.values():
        if len(ids) > 1:
            errors.append(f"E1 duplicate normalized question: {sorted(ids)}")

    indexed_norm: set[str] = set()
    index_toksets: list[frozenset[str]] = []
    for q in indexed_questions:
        indexed_norm.add(normalize_question(q))
        index_toksets.append(_tokset(q))
    by_token: dict[str, list[int]] = defaultdict(list)
    for i, ts in enumerate(index_toksets):
        for t in ts:
            by_token[t].append(i)

    probe_overlap = 0
    for e in examples:
        want = EVAL_TO_SPLIT[e.eval_split]
        src = list(e.derived_from_record_ids)
        refs = src + list(e.gold_record_ids)
        # E5
        unknown = [r for r in refs if r not in record_group]
        if unknown:
            errors.append(f"E5 {e.example_id}: unknown record ids {unknown[:3]}")
        # E2
        for r in refs:
            g = record_group.get(r)
            if g is not None and group_split.get(g) != want:
                errors.append(f"E2 {e.example_id}: record {r} in split {group_split.get(g)!r}, expected {want!r}")
        if e.split_group_id is not None and src:
            src_groups = groups_of(src)
            if src_groups and e.split_group_id not in src_groups:
                errors.append(f"E2 {e.example_id}: split_group_id does not match its source records")
        if e.split_group_id is not None and e.split_group_id in group_split and group_split[e.split_group_id] != want:
            errors.append(f"E2 {e.example_id}: split_group_id in split {group_split[e.split_group_id]!r}")
        # E3 / near-copy
        norm = normalize_question(e.question)
        if norm in indexed_norm:
            if e.eval_split == "train_probe":
                warnings.append(f"W {e.example_id}: train_probe reuses an indexed question verbatim")
            else:
                errors.append(f"E3 {e.example_id}: question equals an indexed question after normalization")
        elif e.eval_split != "train_probe":
            ts = _tokset(e.question)
            cand: Counter[int] = Counter(i for t in ts for i in by_token.get(t, ()))
            for i, inter in cand.items():
                union = len(ts | index_toksets[i])
                if union and inter / union >= near_copy_jaccard:
                    warnings.append(f"W {e.example_id}: near-copy of an indexed question (J={inter / union:.2f})")
                    break
        # E4
        item_groups = groups_of(refs) | ({e.split_group_id} if e.split_group_id else set())
        if e.eval_split == "train_probe":
            probe_overlap += bool(item_groups & sft_g)
            continue
        if item_groups & sft_g:
            errors.append(f"E4 {e.example_id}: overlaps SFT training groups")
        if item_groups & clf_g:
            errors.append(f"E4 {e.example_id}: overlaps classifier training groups")
        if e.eval_split == "test" and item_groups & thr_g:
            errors.append(f"E4 {e.example_id}: test item overlaps threshold/validation-fitting groups")

    report = LeakageReport(
        n_examples=len(examples),
        counts_by_eval_split=dict(Counter(e.eval_split for e in examples)),
        errors=errors,
        warnings=warnings,
        train_probe_sft_overlap=probe_overlap,
        near_copy_jaccard=near_copy_jaccard,
    )
    if fail and errors:
        preview = "; ".join(errors[:5])
        raise LeakageError(f"{len(errors)} leakage error(s): {preview}")
    return report
