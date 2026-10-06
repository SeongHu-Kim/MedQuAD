"""Quick DEV-only retrieval diagnostics for tuning (D-018). Official scoring is evaluation-safety-engineer's.

Usage: python scripts/retrieval/score_dev_runs.py artifacts/evaluation/evalsets/dev.jsonl RUN.jsonl [RUN.jsonl ...]
Scores only items with non-empty gold_record_ids. Refuses any query file whose name does not contain 'dev'.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

KS = (1, 5, 10, 20)


def main(argv: list[str]) -> int:
    queries, runs = Path(argv[0]), [Path(p) for p in argv[1:]]
    if "dev" not in queries.name:
        print("error: DEV query files only (D-018/D-035)", file=sys.stderr)
        return 2
    gold = {}
    for line in queries.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item.get("gold_record_ids"):
            gold[item["example_id"]] = set(item["gold_record_ids"])
    print(f"scored items: {len(gold)}")
    print(f"{'run':<40}" + "".join(f"R@{k:<6}" for k in KS) + "MRR@20")
    for run in runs:
        rows = {r["example_id"]: r for r in map(json.loads, run.read_text(encoding="utf-8").splitlines())}
        recall = dict.fromkeys(KS, 0.0)
        mrr = 0.0
        for eid, g in gold.items():
            ranked = [h["record_id"] for h in rows[eid]["hits"]]
            first = next((i for i, rid in enumerate(ranked, start=1) if rid in g), None)
            for k in KS:
                recall[k] += 1.0 if first is not None and first <= k else 0.0
            mrr += 1.0 / first if first is not None and first <= 20 else 0.0
        n = len(gold)
        print(f"{run.stem:<40}" + "".join(f"{recall[k] / n:<8.3f}" for k in KS) + f"{mrr / n:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
