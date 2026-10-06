"""Independent recomputation of answerability-classifier metrics from IDs-only score files (no TF import).

Usage: python scripts/evaluation/score_classifier.py --model-dir artifacts/models/answerability/<model> [...]
          --out artifacts/evaluation/classifier/metrics.json
Checks every scored pair_id against pairs_manifest_<split>.jsonl (label agreement, split), reports test metrics
with and without own-answer positives, per negative type, and the validation metrics, at the frozen threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from medquad_qa.evaluation.classifier_metrics import classifier_report, reliability_bins
from medquad_qa.evaluation.evalset import sha256_file

ROOT = Path(__file__).resolve().parents[2]
PAIRS = ROOT / "artifacts/models/answerability/pairs"


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def report(rows: list[dict[str, Any]], thr: float) -> dict[str, Any]:
    r = classifier_report([int(x["label"]) for x in rows], [float(x["score"]) for x in rows], thr)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", action="append", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    manifests = {
        s: {r["pair_id"]: r for r in read(PAIRS / f"pairs_manifest_{s}.jsonl")} for s in ("validation", "test")
    }
    out: dict[str, Any] = {"models": {}}
    for d in map(Path, args.model_dir):
        side = json.loads((d / "scores_threshold.json").read_text())
        thr = float(side["threshold"])
        m: dict[str, Any] = {"threshold": thr, "threshold_sidecar": side, "splits": {}}
        for split in ("validation", "test"):
            f = d / f"scores_{split}.jsonl"
            rows = read(f)
            man = manifests[split]
            missing = [r["pair_id"] for r in rows if r["pair_id"] not in man]
            mismatch = [
                r["pair_id"]
                for r in rows
                if r["pair_id"] in man and bool(man[r["pair_id"]]["label"]) != bool(r["label"])
            ]
            dup = len(rows) - len({r["pair_id"] for r in rows})
            not_scored = len(set(man) - {r["pair_id"] for r in rows})
            no_own = [r for r in rows if not r.get("own_answer")]
            by_neg: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for r in no_own:
                if not r["label"]:
                    by_neg[r.get("negative_type") or "none"].append(r)
            pos = [r for r in no_own if r["label"]]
            m["splits"][split] = {
                "file": str(f),
                "sha256": sha256_file(f),
                "integrity": {
                    "n": len(rows),
                    "missing_in_manifest": len(missing),
                    "label_mismatch": len(mismatch),
                    "duplicates": dup,
                    "manifest_pairs_not_scored": not_scored,
                },
                "all_pairs": report(rows, thr),
                "excluding_own_answer": report(no_own, thr),
                # positives vs one negative type at a time (how hard each negative family is)
                "by_negative_type": {k: report(pos + v, thr) for k, v in sorted(by_neg.items())},
                "reliability_excluding_own_answer": reliability_bins(
                    [int(r["label"]) for r in no_own], [float(r["score"]) for r in no_own]
                ),
            }
        out["models"][d.name] = m
        t = m["splits"]["test"]["excluding_own_answer"]
        print(
            d.name,
            {
                k: (round(t[k], 4) if isinstance(t[k], float) else t[k])
                for k in ("n", "auroc", "auprc", "brier", "ece", "f1")
            },
            m["splits"]["test"]["integrity"],
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
