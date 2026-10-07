"""Verify E4 Step 0 (D-048) outputs and project the TEST runtime.

Usage: python scripts/evaluation/check_step0.py [--dir artifacts/evaluation/e4/step0_dev]
       [--pre-resume <snapshot of per_item/dev__base.jsonl taken right after the kill>]
Writes <dir>/step0_check.json (IDs, counts, versions, timings only).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from medquad_qa.contracts import RAG_MODES, QAResponse
from medquad_qa.evaluation.claims import extract_claims
from medquad_qa.evaluation.evalset import load_eval_set

ROOT = Path(__file__).resolve().parents[2]
MODES = ("base", "rag", "finetuned", "finetuned_rag")
# TEST item counts per mode in the approved plan (D-048), split by whether the item can skip generation.
PLAN = {
    "base": {"track_a": 300, "probe": 100, "c_personal": 40, "c_control": 40},
    "finetuned": {"track_a": 300, "probe": 100, "c_personal": 40, "c_control": 40},
    "rag": {"track_a": 300, "c_all": 240},
    "finetuned_rag": {"track_a": 300, "c_all": 240},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "artifacts/evaluation/e4/step0_dev"))
    ap.add_argument("--pre-resume", default=None)
    args = ap.parse_args()
    d = Path(args.dir)
    ex = {e.example_id: e for e in load_eval_set(ROOT / "artifacts/evaluation/evalsets/dev.jsonl")}
    out: dict[str, Any] = {"modes": {}}
    ok = True
    timings: dict[str, dict[str, float]] = {}
    for mode in MODES:
        path = d / "per_item" / f"dev__{mode}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        ids = [r["example_id"] for r in rows]
        resps = []
        schema_errors = 0
        for r in rows:
            try:
                resps.append(QAResponse.model_validate(r["response"]))
            except Exception:  # noqa: BLE001 - counted and reported
                schema_errors += 1
        prov_ok = all(
            r["label_provenance"] == ex[r["example_id"]].label_provenance
            and r["question_provenance"] == ex[r["example_id"]].question_provenance
            for r in rows
        )
        subset = all(set(c.record_id for c in x.citations) <= set(x.retrieved_record_ids) for x in resps)
        answered = [x for x in resps if not x.abstained]
        claims = [c for x in answered for c in extract_claims(x.answer)]
        cited = [c for c in claims if c.cited_ids]
        gen = [r["item_wall_ms"] / 1000 for r in rows if r.get("response") and r["response"]["generation"]]
        early = [r["item_wall_ms"] / 1000 for r in rows if r.get("response") and not r["response"]["generation"]]
        timings[mode] = {
            "gen_mean_s": statistics.mean(gen) if gen else 0.0,
            "early_mean_s": statistics.mean(early) if early else 0.0,
        }
        meta = json.loads((d / f"dev__{mode}.meta.json").read_text())
        m = {
            "n_rows": len(rows),
            "unique_ids": len(set(ids)),
            "schema_valid": len(resps),
            "schema_errors": schema_errors,
            "error_rows": sum("error" in r for r in rows),
            "model_versions": sorted({x.model_version for x in resps}),
            "prompt_versions": sorted({x.prompt_version for x in resps}),
            "retrievers": dict(Counter(x.retriever for x in resps)),
            "lexical_fallback": sum("lexical_fallback" in x.warnings for x in resps),
            "warnings": dict(Counter(w for x in resps for w in x.warnings)),
            "citations_subset_of_retrieved": subset,
            "invalid_citation_ids_total": sum(len(x.invalid_citation_ids) for x in resps),
            "citation_coverage": len(cited) / len(claims) if claims else None,
            "abstention_reasons": dict(Counter(x.abstention_reason or "answered" for x in resps)),
            "label_provenance_preserved": prov_ok,
            "item_wall_s": {
                "median_all": statistics.median(r["item_wall_ms"] / 1000 for r in rows),
                "mean_generated": timings[mode]["gen_mean_s"],
                "n_generated": len(gen),
                "mean_no_generation": timings[mode]["early_mean_s"],
                "n_no_generation": len(early),
            },
            "meta": {
                k: meta.get(k)
                for k in (
                    "git_sha",
                    "versions",
                    "n_resumed_rows",
                    "n_new_rows_this_session",
                    "n_rows_total",
                    "evalset_sha256",
                    "evalset_manifest_sha256",
                )
            },
        }
        if mode in RAG_MODES:
            m["missing_citations_rate"] = sum(x.abstention_reason == "missing_citations" for x in resps) / len(resps)
        ok &= schema_errors == 0 and len(set(ids)) == len(ids) and prov_ok and subset and m["lexical_fallback"] == 0
        out["modes"][mode] = m
    if args.pre_resume:
        before = Path(args.pre_resume).read_text(encoding="utf-8").splitlines()
        after = (d / "per_item" / "dev__base.jsonl").read_text(encoding="utf-8").splitlines()
        out["resume_test"] = {
            "rows_before_kill": len(before),
            "prefix_byte_identical": after[: len(before)] == before,
            "rows_after_resume": len(after),
            "duplicated_ids": len(after) - len({json.loads(x)["example_id"] for x in after}),
        }
        ok &= out["resume_test"]["prefix_byte_identical"] and out["resume_test"]["duplicated_ids"] == 0
    # Projection: generated items at the measured per-mode mean; personal items (refused pre-generation) at the
    # no-generation mean; other items assumed to generate (upper bound: early gate abstentions are faster).
    proj = {}
    for mode, parts in PLAN.items():
        t = timings[mode]
        gen_items = sum(v for k, v in parts.items() if k != "c_personal")
        proj[mode] = gen_items * t["gen_mean_s"] + parts.get("c_personal", 0) * t["early_mean_s"]
    out["projection_s"] = {**proj, "total": sum(proj.values()), "total_hours": sum(proj.values()) / 3600}
    out["passed"] = bool(ok)
    (d / "step0_check.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps(out, indent=1, default=str)[:6000])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
