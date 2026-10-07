"""Write the offline-evaluation gauge summary (artifacts/evaluation/summary/latest_offline_eval.json).

Reproduces the command used on 2026-10-07 to build the E4 summary from saved metrics only (no model calls):
  python scripts/evaluation/write_offline_summary.py --eval-run-id e4-1c84b60-20261006 \
      [--metrics artifacts/evaluation/e4/metrics.json] \
      [--retrieval artifacts/evaluation/retrieval/scores_test_track_a.json] \
      [--classifier artifacts/evaluation/classifier/metrics.json] \
      [--out artifacts/evaluation/summary/latest_offline_eval.json]
Format: medquad_qa.evaluation.summary (agreed with service-platform-engineer); written atomically.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from medquad_qa.evaluation.summary import build_summary, write_summary_atomic

ROOT = Path(__file__).resolve().parents[2]
TRACK_LABEL = {"test_track_a": "B", "test_track_c": "C", "train_probe": "B_probe"}


def rows_from(metrics: dict[str, Any], retrieval: dict[str, Any], classifier: dict[str, Any]) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for name, r in retrieval["runs"].items():
        for k in ("recall@1", "recall@5", "mrr@20"):
            rows.append(("A", name, k, r["metrics"][k]))
    for s, sv in metrics["sets"].items():
        tr = TRACK_LABEL[s]
        for mode, m in sv["modes"].items():
            a = m["summary"]["abstention"]["all"]
            lat = m["item_wall_latency"] or {}
            rows += [
                (tr, mode, "over_refusal_rate", a["over_refusal_rate"]),
                (tr, mode, "abstain_recall", a["abstain_recall"]),
                (tr, mode, "ref_coverage", m["reference"]["ref_coverage_mean_all"]),
                (tr, mode, "ref_unsupported_claim_rate", m["reference"]["ref_unsupported_claim_rate_mean"]),
                (tr, mode, "p95_latency_ms", lat.get("p95_ms")),
                (tr, mode, "median_latency_ms", lat.get("median_ms")),
            ]
            c = m.get("citations")
            if c:
                rows += [
                    (tr, mode, "citation_validity", c["citation_validity"]),
                    (tr, mode, "citation_support", c["citation_support"]),
                    (tr, mode, "unsupported_claim_rate", c["unsupported_claim_rate"]),
                ]
            if m.get("injection"):
                inj = m["injection"]
                rows.append((tr, mode, "injection_canary_leak_rate", inj["canary_leaked"] / inj["n"]))
    for model, d in classifier["models"].items():
        t = d["splits"]["test"]["excluding_own_answer"]
        for k in ("auroc", "auprc", "brier", "ece"):
            rows.append(("classifier", model, k, t[k]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-run-id", required=True)
    ap.add_argument("--metrics", default=str(ROOT / "artifacts/evaluation/e4/metrics.json"))
    ap.add_argument("--retrieval", default=str(ROOT / "artifacts/evaluation/retrieval/scores_test_track_a.json"))
    ap.add_argument("--classifier", default=str(ROOT / "artifacts/evaluation/classifier/metrics.json"))
    ap.add_argument("--out", default=str(ROOT / "artifacts/evaluation/summary/latest_offline_eval.json"))
    args = ap.parse_args()
    load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))  # noqa: E731
    summary = build_summary(
        args.eval_run_id, rows_from(load(args.metrics), load(args.retrieval), load(args.classifier))
    )
    write_summary_atomic(summary, args.out)
    print(len(summary["metrics"]), "rows ->", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
