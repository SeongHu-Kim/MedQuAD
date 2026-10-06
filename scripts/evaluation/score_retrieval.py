"""Score D-024 retrieval run files against a frozen eval set.

Usage: python scripts/evaluation/score_retrieval.py --evalset SET.jsonl --run RUN.jsonl [--run ...] --out OUT.json
Queries without gold records are excluded and counted. Rows with ``error`` or ``lexical_fallback`` are counted and
scored as returned (an error row has no hits, i.e. a miss).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from medquad_qa.evaluation import latency
from medquad_qa.evaluation.evalset import load_eval_set, sha256_file
from medquad_qa.evaluation.retrieval_metrics import aggregate, score_query
from medquad_qa.evaluation.run_format import load_run
from medquad_qa.evaluation.stats import paired_cluster_bootstrap

KS = (1, 5, 10, 20)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evalset", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="unrecorded", help="device the runs used (the run meta does not record it)")
    ap.add_argument("--latency-valid", action="store_true", help="runs did not overlap other GPU/CPU-heavy jobs")
    args = ap.parse_args()
    examples = load_eval_set(args.evalset)
    scorable = [e for e in examples if e.gold_record_ids]
    by_id = {e.example_id: e for e in examples}
    result: dict[str, object] = {
        "evalset": args.evalset,
        "evalset_sha256": sha256_file(args.evalset),
        "n_examples": len(examples),
        "n_scorable": len(scorable),
        "device": args.device,
        "latency_valid": args.latency_valid,
        "runs": {},
    }
    per_query: dict[str, dict[str, float]] = {}
    for path in args.run:
        rows = load_run(path, expected_example_ids=by_id)
        meta_path = Path(f"{path}.meta.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        rmap = {r.example_id: r for r in rows}
        scores = [
            score_query(e.example_id, rmap[e.example_id].ranked_ids(), set(e.gold_record_ids), ks=KS) for e in scorable
        ]
        retrievers = Counter(r.retriever for r in rows)
        name = max(retrievers, key=lambda k: retrievers[k])
        per_query[name] = {s.example_id: s.hit_at[5] for s in scores}
        result["runs"][name] = {  # type: ignore[index]
            "run_file": path,
            "run_sha256": sha256_file(path),
            "meta": {k: meta.get(k) for k in ("index_version", "corpus_version", "git_sha", "config", "command")},
            "retrievers_seen": dict(retrievers),
            "n_error_rows": sum(r.error is not None for r in rows),
            "n_lexical_fallback": sum("lexical_fallback" in r.warnings for r in rows),
            "metrics": aggregate(scores),
            "latency": latency.summarize([r.latency_ms for r in rows]),
        }
    names = sorted(per_query)
    clusters = [by_id[i].split_group_id or i for i in sorted(per_query[names[0]])] if names else []
    comparisons = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ids = sorted(per_query[a])
            comparisons[f"{b} - {a} (recall@5)"] = paired_cluster_bootstrap(
                [per_query[a][x] for x in ids], [per_query[b][x] for x in ids], clusters
            )
    result["paired_comparisons"] = comparisons
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for name in names:
        print(name, json.dumps(result["runs"][name]["metrics"]))  # type: ignore[index]
    return 0


if __name__ == "__main__":
    sys.exit(main())
