"""Sequential latency measurement through the running HTTP API (Docker stack or host process).

Usage:
  python scripts/evaluation/api_latency.py --evalset artifacts/evaluation/evalsets/dev.jsonl --n 100 \
      --modes rag base --base-url http://127.0.0.1:8000 --out artifacts/evaluation/latency/<name>.json

Queries cycle through the eval set in file order until ``--n`` requests per mode have been sent (repeats are
counted and reported). Records per request: HTTP status, client wall time, server latency_ms, component
latencies, abstention flag/reason and warnings. No question or answer text is stored. Only DEV may be used here
(D-035); TEST sets are refused.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from medquad_qa.evaluation import latency
from medquad_qa.evaluation.evalset import load_eval_set, sha256_file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evalset", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--modes", nargs="+", default=["rag", "base"])
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--label", default="docker")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    examples = load_eval_set(args.evalset)
    if any(e.eval_split != "dev" for e in examples):
        print("refusing: latency runs use DEV queries only (D-035)", file=sys.stderr)
        return 2
    client = httpx.Client(base_url=args.base_url, timeout=180.0)
    info = client.get("/v1/info").json()
    result: dict[str, Any] = {
        "label": args.label,
        "base_url": args.base_url,
        "evalset": args.evalset,
        "evalset_sha256": sha256_file(args.evalset),
        "n_unique_queries": len(examples),
        "requests_per_mode": args.n,
        "top_k": args.top_k,
        "conditions": "sequential, one client, no concurrency; server has one generation slot",
        "server_info": info,
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "modes": {},
    }
    for mode in args.modes:
        rows = []
        for i in range(args.n):
            e = examples[i % len(examples)]
            rid = f"e4-lat-{args.label}-{mode}-{i:03d}"
            t = time.perf_counter()
            r = client.post(
                "/v1/qa",
                json={"question": e.question, "experiment_mode": mode, "top_k": args.top_k},
                headers={"X-Request-ID": rid},
            )
            wall = (time.perf_counter() - t) * 1000
            row: dict[str, Any] = {
                "request_id": rid,
                "example_id": e.example_id,
                "status": r.status_code,
                "wall_ms": wall,
            }
            if r.status_code == 200:
                body = r.json()
                row.update(
                    latency_ms=body["latency_ms"],
                    component_latency_ms=body["component_latency_ms"],
                    abstained=body["abstained"],
                    abstention_reason=body["abstention_reason"],
                    warnings=body["warnings"],
                    completion_tokens=(body.get("generation") or {}).get("completion_tokens"),
                )
            else:
                row["error_code"] = r.json().get("error_code")
            rows.append(row)
            print(mode, i, r.status_code, round(wall), flush=True)
        ok = [x for x in rows if x["status"] == 200]
        result["modes"][mode] = {
            "n": len(rows),
            "n_ok": len(ok),
            "status_counts": {str(s): sum(x["status"] == s for x in rows) for s in sorted({x["status"] for x in rows})},
            "client_wall": latency.summarize([x["wall_ms"] for x in rows]),
            "server_latency": latency.summarize([x["latency_ms"] for x in ok]) if ok else None,
            "server_latency_answered": latency.summarize([x["latency_ms"] for x in ok if not x["abstained"]])
            if any(not x["abstained"] for x in ok)
            else None,
            "component_latency": latency.summarize_components([x["component_latency_ms"] for x in ok]),
            "abstained": sum(bool(x.get("abstained")) for x in ok),
            "rows": rows,
        }
    result["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for mode, m in result["modes"].items():
        print(mode, json.dumps({k: m[k] for k in ("n", "n_ok", "status_counts", "client_wall", "server_latency")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
