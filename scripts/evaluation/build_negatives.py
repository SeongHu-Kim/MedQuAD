"""Build the train/validation negatives file for the F-009/F-010 remediation (D-060).

Contents: normalized questions and answer sentences (>= 4 words) of TRAIN and VALIDATION records, excluding every
split group that contains a train_probe item. No test-split records. Used only to measure the safety detector's
over-refusal and the evidence filter's false-flag rate. Rows are text only (no record or group IDs) and are shuffled
with a fixed seed, so neither the excluded groups nor corpus order can be read from the file.

Usage: python scripts/evaluation/build_negatives.py [--out artifacts/evaluation/negatives_v1/negatives.jsonl]
Prints totals only. The output directory is git-ignored (artifacts/evaluation/negatives_*/).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from medquad_qa.evaluation.claims import split_sentences
from medquad_qa.evaluation.evalset import load_eval_set

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261008
MIN_WORDS = 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artifacts/evaluation/negatives_v1/negatives.jsonl"))
    args = ap.parse_args()
    split_of: dict[str, str] = {}
    group_of: dict[str, str] = {}
    with (ROOT / "data/manifests/split_manifest.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            m = json.loads(line)
            split_of[m["record_id"]], group_of[m["record_id"]] = m["split"], m["split_group_id"]
    excluded: set[str] = set()
    for e in load_eval_set(ROOT / "artifacts/evaluation/evalsets/train_probe.jsonl"):
        if e.split_group_id:
            excluded.add(e.split_group_id)
        for rid in [*e.gold_record_ids, *e.derived_from_record_ids]:
            if rid in group_of:
                excluded.add(group_of[rid])
    rows: list[dict[str, str]] = []
    records = groups = 0
    seen_groups: set[str] = set()
    with (ROOT / "data/processed/corpus.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            rid = d["record_id"]
            if split_of.get(rid) not in ("train", "validation") or group_of[rid] in excluded:
                continue
            records += 1
            seen_groups.add(group_of[rid])
            rows.append({"kind": "question", "split": split_of[rid], "text": d["question"]})
            for s in split_sentences(d["answer"]):
                if len(s.split()) >= MIN_WORDS:
                    rows.append({"kind": "answer_sentence", "split": split_of[rid], "text": s})
    groups = len(seen_groups)
    random.Random(SEED).shuffle(rows)  # noqa: S311 - reproducible order, not security-relevant
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    totals = {
        "records": records,
        "groups": groups,
        "questions": sum(r["kind"] == "question" for r in rows),
        "answer_sentences": sum(r["kind"] == "answer_sentence" for r in rows),
        "by_split": {s: sum(r["split"] == s for r in rows) for s in ("train", "validation")},
    }
    print(json.dumps(totals))
    return 0


if __name__ == "__main__":
    sys.exit(main())
