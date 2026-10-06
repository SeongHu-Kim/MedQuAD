"""Frozen evaluation-set I/O and manifest.

Committed eval files hold IDs, (paraphrased) questions and labels only (D-015): ``reference_answer`` must be None
and ``injected_evidence`` may only contain synthetic text written for fixtures. Reference answers are resolved
from the corpus at runtime. The manifest pins the file sha256 so any later edit is detected.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from medquad_qa.contracts.evaluation import EvaluationExample

EVALSET_FORMAT_VERSION = "evalset-v1"
HUMAN_PROVENANCE = frozenset({"human_reviewed", "llm_generated_reviewed"})


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_examples(examples: Sequence[EvaluationExample]) -> None:
    """Policy checks beyond the contract: no dataset text, truthful review provenance."""
    for e in examples:
        if e.reference_answer is not None:
            raise ValueError(f"{e.example_id}: reference_answer must be None in committed eval files (D-015)")
        human = e.label_provenance in HUMAN_PROVENANCE or e.question_provenance in HUMAN_PROVENANCE
        if human != (e.reviewer == "human_nonclinical"):
            raise ValueError(
                f"{e.example_id}: human-reviewed provenance requires reviewer='human_nonclinical' and vice versa"
            )
        if e.answerable and e.expected_behavior == "abstain" and e.case_type != "personalized_advice":
            raise ValueError(f"{e.example_id}: answerable item expected to abstain (only personalized_advice may)")
        if e.expected_behavior == "abstain" and e.expected_abstention_reason is None:
            raise ValueError(f"{e.example_id}: abstain items need expected_abstention_reason")


def write_eval_set(
    examples: Sequence[EvaluationExample], path: str | Path, *, meta: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Write JSONL (sorted keys, input order) plus ``<path>.manifest.json``; return the manifest."""
    validate_examples(examples)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        for e in examples:
            fh.write(json.dumps(e.model_dump(mode="json"), sort_keys=True, ensure_ascii=False) + "\n")
    manifest: dict[str, Any] = {
        "format_version": EVALSET_FORMAT_VERSION,
        "file": p.name,
        "sha256": sha256_file(p),
        "n_examples": len(examples),
        "by_eval_split": dict(Counter(e.eval_split for e in examples)),
        "by_track": dict(Counter(e.evaluation_track for e in examples)),
        "by_case_type": dict(Counter(e.case_type for e in examples)),
        "by_question_provenance": dict(Counter(e.question_provenance for e in examples)),
        "by_label_provenance": dict(Counter(e.label_provenance for e in examples)),
        "by_reviewer": dict(Counter(e.reviewer for e in examples)),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **(meta or {}),
    }
    Path(f"{p}.manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_eval_set(path: str | Path, *, verify: bool = True) -> list[EvaluationExample]:
    p = Path(path)
    if verify:
        manifest = json.loads(Path(f"{p}.manifest.json").read_text(encoding="utf-8"))
        actual = sha256_file(p)
        if manifest["sha256"] != actual:
            raise ValueError(f"{p}: sha256 {actual} does not match frozen manifest {manifest['sha256']}")
    with p.open(encoding="utf-8") as fh:
        examples = [EvaluationExample.model_validate_json(line) for line in fh if line.strip()]
    validate_examples(examples)
    return examples
