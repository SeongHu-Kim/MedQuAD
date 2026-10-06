"""Fit the heuristic answerability-gate threshold on VALIDATION pairs only (D-018).

Usage: python -m medquad_qa.rag.gate_fit --pairs <answerability_pairs_validation.jsonl> \
           --out configs/retrieval/gate_heuristic.json
Input rows are ``AnswerabilityPair`` JSON; any row whose split is not 'validation' is rejected.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from medquad_qa.contracts import AnswerabilityPair
from medquad_qa.rag.gate import HEURISTIC_MODEL_VERSION, coverage, fit_threshold, heuristic_version
from medquad_qa.retrieval.corpus import sha256_file


def fit_from_pairs(pairs_path: Path) -> dict[str, object]:
    scores: list[float] = []
    labels: list[bool] = []
    with pairs_path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            pair = AnswerabilityPair.model_validate_json(line)
            if pair.split != "validation":
                raise ValueError(f"{pairs_path}:{lineno}: split '{pair.split}' not allowed (validation only)")
            scores.append(coverage(pair.question, pair.evidence_text))
            labels.append(pair.label)
    threshold, f1 = fit_threshold(scores, labels)
    data_sha = sha256_file(pairs_path)
    return {
        "threshold": threshold,
        "threshold_version": heuristic_version(threshold, data_sha),
        "model_version": HEURISTIC_MODEL_VERSION,
        "criterion": "max_f1",
        "validation_f1": round(f1, 6),
        "n_pairs": len(scores),
        "n_positive": sum(labels),
        "fitted_on": {"path": str(pairs_path), "sha256": data_sha, "split": "validation"},
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", required=True)
    p.add_argument("--out", default="configs/retrieval/gate_heuristic.json")
    args = p.parse_args(argv)
    try:
        result = fit_from_pairs(Path(args.pairs))
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
