"""E4 gate: run check_leakage over ALL frozen eval sets against the protected training/threshold data.

Usage:
  python scripts/evaluation/check_protected_leakage.py \
      --sft artifacts/models/runs/<run_id>/sft_record_ids.jsonl \
      --pairs-train artifacts/models/answerability/pairs/pairs_manifest_train.jsonl \
      --pairs-validation artifacts/models/answerability/pairs/pairs_manifest_validation.jsonl \
      [--threshold-ids <jsonl with record_id>]... --out artifacts/evaluation/leakage/protected_check.json

Classifier-train = train pairs (question + evidence record IDs); threshold/validation-fitting = validation pairs
plus any extra --threshold-ids files (e.g. the RAG gate fit). All manifests must carry the frozen split_version.
Exit 1 on any error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from medquad_qa.evaluation.evalset import load_eval_set, sha256_file
from medquad_qa.evaluation.leakage import check_leakage

ROOT = Path(__file__).resolve().parents[2]
SETS = ("test_track_a", "test_track_c", "train_probe", "dev")


def read_ids(path: str, fields: tuple[str, ...], split_version: str) -> tuple[set[str], int]:
    ids: set[str] = set()
    rows = 0
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            sv = row.get("split_version")
            if sv is not None and sv != split_version:
                raise ValueError(f"{path}: split_version {sv} != frozen {split_version}")
            for f in fields:
                if row.get(f):
                    ids.add(row[f])
    return ids, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True)
    ap.add_argument("--pairs-train", required=True)
    ap.add_argument("--pairs-validation", required=True)
    ap.add_argument("--threshold-ids", action="append", default=[])
    ap.add_argument("--evalsets", default=str(ROOT / "artifacts/evaluation/evalsets"))
    ap.add_argument("--out", default=str(ROOT / "artifacts/evaluation/leakage/protected_check.json"))
    args = ap.parse_args()

    meta = json.loads((ROOT / "data/manifests/split_manifest.meta.json").read_text(encoding="utf-8"))
    sv = meta["split_version"]
    record_group: dict[str, str] = {}
    group_split: dict[str, str] = {}
    with (ROOT / "data/manifests/split_manifest.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            m = json.loads(line)
            record_group[m["record_id"]] = m["split_group_id"]
            group_split[m["split_group_id"]] = m["split"]
    questions = [json.loads(line)["question"] for line in (ROOT / "data/processed/corpus.jsonl").open(encoding="utf-8")]

    sft, n_sft = read_ids(args.sft, ("record_id",), sv)
    clf, n_clf = read_ids(args.pairs_train, ("question_record_id", "evidence_record_id"), sv)
    thr, n_thr = read_ids(args.pairs_validation, ("question_record_id", "evidence_record_id"), sv)
    for p in args.threshold_ids:
        extra, _ = read_ids(p, ("record_id", "question_record_id", "evidence_record_id"), sv)
        thr |= extra

    # Sanity: protected sets must come from the splits they claim.
    wrong = {
        "sft_not_train": sorted(r for r in sft if group_split.get(record_group.get(r, "")) != "train")[:10],
        "clf_not_train": sorted(r for r in clf if group_split.get(record_group.get(r, "")) != "train")[:10],
        "thr_not_validation": sorted(r for r in thr if group_split.get(record_group.get(r, "")) != "validation")[:10],
    }
    examples = [e for name in SETS for e in load_eval_set(Path(args.evalsets) / f"{name}.jsonl")]
    report = check_leakage(
        examples,
        record_group,
        group_split,  # type: ignore[arg-type]
        questions,
        sft_record_ids=sft,
        classifier_train_record_ids=clf,
        threshold_record_ids=thr,
        fail=False,
    )
    errors = list(report.errors) + [f"{k}: {v}" for k, v in wrong.items() if v]
    out = {
        "split_version": sv,
        "inputs": {p: sha256_file(p) for p in [args.sft, args.pairs_train, args.pairs_validation, *args.threshold_ids]},
        "rows": {"sft": n_sft, "pairs_train": n_clf, "pairs_validation": n_thr},
        "protected_record_ids": {"sft": len(sft), "classifier_train": len(clf), "threshold": len(thr)},
        "n_eval_examples": len(examples),
        "train_probe_sft_overlap": report.train_probe_sft_overlap,
        "errors": errors,
        "warnings": report.warnings,
        "passed": not errors,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("passed", "rows", "protected_record_ids", "train_probe_sft_overlap")}))
    for e in errors[:20]:
        print("  ", e)
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
