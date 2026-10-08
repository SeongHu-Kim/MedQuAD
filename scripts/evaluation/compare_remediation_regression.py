"""f4bf7e9 §3 four-mode regression report: remediated pipeline vs E4, from stored outputs only.

Usage: python scripts/evaluation/compare_remediation_regression.py \
    [--e4 artifacts/evaluation/e4] [--new artifacts/evaluation/remediation/regression] \
    [--out artifacts/evaluation/remediation/regression/regression_check.json]
Refuses (exit 2) unless both metrics.json provenance blocks match (same score_e4.py, qa_report.py, metrics config,
corpus, NLI). Output: counts, CIs and per-cell regression verdicts only; any regression fails the remediation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from medquad_qa.evaluation import regression_check as rc
from medquad_qa.evaluation.e4b_compare import check_provenance, load_item_scores, load_responses
from medquad_qa.evaluation.evalset import load_eval_set, sha256_file

ROOT = Path(__file__).resolve().parents[2]
MODES = ("base", "rag", "finetuned", "finetuned_rag")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e4", default=str(ROOT / "artifacts/evaluation/e4"))
    ap.add_argument("--new", default=str(ROOT / "artifacts/evaluation/remediation/regression"))
    ap.add_argument("--evalsets", default=str(ROOT / "artifacts/evaluation/evalsets"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    e4, new = Path(args.e4), Path(args.new)
    m4 = json.loads((e4 / "metrics.json").read_text())
    mn = json.loads((new / "metrics.json").read_text())
    bad = check_provenance(m4, mn)
    if bad:
        print(f"provenance mismatch, refusing: {bad}", file=sys.stderr)
        return 2

    def ex(name: str) -> dict[str, Any]:
        return {e.example_id: e for e in load_eval_set(Path(args.evalsets) / f"{name}.jsonl")}

    sets = {"test_track_a": ex("test_track_a"), "test_track_c": ex("test_track_c"), "train_probe": ex("train_probe")}
    s_old = load_item_scores(e4 / "per_item" / "item_scores.jsonl")
    s_new = load_item_scores(new / "per_item" / "item_scores.jsonl")

    def resp(tree: Path, s: str, mode: str) -> Any:
        p = tree / "runs" / "per_item" / f"{s}__{mode}.jsonl"
        return load_responses(p) if p.exists() else None

    out: dict[str, Any] = {
        "provenance": {
            "e4_metrics_sha256": sha256_file(e4 / "metrics.json"),
            "new_metrics_sha256": sha256_file(new / "metrics.json"),
            "new_versions": {k: v for k, v in (mn["sets"]["test_track_a"]["modes"]["rag"]["versions"] or {}).items()},
        },
        "track_a": {},
        "track_c": {},
        "train_probe": {},
    }
    for mode in MODES:
        o, n = resp(e4, "test_track_a", mode), resp(new, "test_track_a", mode)
        if o is not None and n is not None:
            out["track_a"][mode] = rc.track_a(
                mode,
                sets["test_track_a"],
                o,
                n,
                s_old.get(("test_track_a", mode), {}),
                s_new.get(("test_track_a", mode), {}),
                mn["sets"]["test_track_a"]["modes"].get(mode, {}).get("citations"),
            )
        o, n = resp(e4, "test_track_c", mode), resp(new, "test_track_c", mode)
        if o is not None and n is not None:
            out["track_c"][mode] = rc.track_c(mode, sets["test_track_c"], o, n)
    out["train_probe"]["finetuned"] = rc.train_probe(
        sets["train_probe"], s_old.get(("train_probe", "finetuned"), {}), s_new.get(("train_probe", "finetuned"), {})
    )
    out["any_regression"] = rc.any_regression({k: out[k] for k in ("track_a", "track_c", "train_probe")})
    dest = Path(args.out) if args.out else new / "regression_check.json"
    dest.write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print("any_regression", out["any_regression"], "->", dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
