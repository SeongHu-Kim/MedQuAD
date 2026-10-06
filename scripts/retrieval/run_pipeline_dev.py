"""Run the real RAG pipeline on DEV queries for integration evidence and DEV-only tuning (D-018).

Usage: python scripts/retrieval/run_pipeline_dev.py --queries artifacts/evaluation/evalsets/dev.jsonl \
           --out artifacts/indexes/runs/dev/pipeline_rag.jsonl [--mode rag] [--limit N]
Output rows hold answers (gitignored location); the summary printed at the end holds counts only.
Settings come from MEDQUAD_* environment variables (see docs/retrieval/README.md).
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

from medquad_qa.contracts import MedQuADError, QARequest
from medquad_qa.rag.factory import build_pipeline
from medquad_qa.rag.settings import RagSettings


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queries", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", default="rag", choices=["base", "rag", "finetuned", "finetuned_rag"])
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args(argv)
    if "dev" not in Path(args.queries).name:
        print("error: DEV query files only (D-018/D-035)", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    pipe = build_pipeline(RagSettings.from_env(preload_generators=("base",) if "finetuned" not in args.mode else ()))
    build_s = time.perf_counter() - t0
    items = [json.loads(line) for line in Path(args.queries).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        items = items[: args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    reasons: collections.Counter[str] = collections.Counter()
    by_case: collections.Counter[tuple[str, str]] = collections.Counter()
    warnings: collections.Counter[str] = collections.Counter()
    latencies: list[float] = []
    with out.open("w", encoding="utf-8") as fh:
        for item in items:
            req = QARequest(question=item["question"], experiment_mode=args.mode, top_k=args.top_k)
            try:
                r = pipe.answer(req, f"dev-{item['example_id']}")
            except MedQuADError as exc:
                fh.write(json.dumps({"example_id": item["example_id"], "error": type(exc).__name__}) + "\n")
                reasons[f"error:{type(exc).__name__}"] += 1
                continue
            latencies.append(r.latency_ms)
            reason = r.abstention_reason or "answered"
            reasons[reason] += 1
            by_case[(item["case_type"], reason)] += 1
            warnings.update(r.warnings)
            row = {"example_id": item["example_id"], "case_type": item["case_type"], **r.model_dump(mode="json")}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    latencies.sort()
    summary = {
        "queries": len(items),
        "mode": args.mode,
        "build_pipeline_seconds": round(build_s, 1),
        "versions": pipe.versions(),
        "outcomes": dict(reasons),
        "by_case_type": {f"{c}|{r}": n for (c, r), n in sorted(by_case.items())},
        "warnings": dict(warnings),
        "latency_ms_median": latencies[len(latencies) // 2] if latencies else None,
        "latency_ms_max": latencies[-1] if latencies else None,
    }
    print(json.dumps(summary, indent=2))
    out.with_name(out.name + ".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    pipe.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
