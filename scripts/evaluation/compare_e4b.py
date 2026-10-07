"""E4b comparison report (pre-declaration b502e60 + amendment 1): v2c vs E4 rag / v1, from stored outputs only.

Usage (after BOTH trees are scored by the same score_e4.py; source scripts/evaluation/e4b_env.sh first so the frozen
dense:qa retriever is rebuilt to re-derive the supplied evidence chunks for the copy rate):
  CUDA_VISIBLE_DEVICES="" python scripts/evaluation/compare_e4b.py \
      --sft-manifest artifacts/models/runs/sft-main-20261007-104043-4c946221/sft_record_ids.jsonl \
      [--e4 artifacts/evaluation/e4] [--e4b artifacts/evaluation/e4b] [--out artifacts/evaluation/e4b/comparison.json]
Refuses (exit 3) if the retriever is not the frozen dense:qa, and (exit 2) if the two metrics.json provenance
blocks differ (score_e4.py, qa_report.py, metrics config, corpus,
NLI model/threshold). Output: aggregates only, no question/answer text, no IDs.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from medquad_qa.evaluation import e4b_compare as cmp
from medquad_qa.evaluation.evalset import load_eval_set, sha256_file

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_RETRIEVER = "dense:qa"  # D-037 frozen RAG retriever (run with scripts/evaluation/e4b_env.sh sourced)
TOP_K = 5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e4", default=str(ROOT / "artifacts/evaluation/e4"))
    ap.add_argument("--e4b", default=str(ROOT / "artifacts/evaluation/e4b"))
    ap.add_argument("--evalsets", default=str(ROOT / "artifacts/evaluation/evalsets"))
    ap.add_argument("--sft-manifest", required=True)
    ap.add_argument("--out", default=str(ROOT / "artifacts/evaluation/e4b/comparison.json"))
    args = ap.parse_args()
    e4, e4b = Path(args.e4), Path(args.e4b)
    m4 = json.loads((e4 / "metrics.json").read_text())
    m4b = json.loads((e4b / "metrics.json").read_text())
    bad = cmp.check_provenance(m4, m4b)
    if bad:
        print(f"provenance mismatch, refusing to compare: {bad}", file=sys.stderr)
        return 2

    def ex(name: str) -> dict[str, cmp.EvaluationExample]:
        return {e.example_id: e for e in load_eval_set(Path(args.evalsets) / f"{name}.jsonl")}

    a_ex, c_ex, p_ex = ex("test_track_a"), ex("test_track_c"), ex("train_probe")

    def resp(tree: Path, s: str, mode: str) -> cmp.Responses:
        return cmp.load_responses(tree / "runs" / "per_item" / f"{s}__{mode}.jsonl")

    rag_a, v1_a, v2_a = (
        resp(e4, "test_track_a", "rag"),
        resp(e4, "test_track_a", "finetuned_rag"),
        resp(e4b, "test_track_a", "finetuned_rag"),
    )
    rag_c, v1_c, v2_c = (
        resp(e4, "test_track_c", "rag"),
        resp(e4, "test_track_c", "finetuned_rag"),
        resp(e4b, "test_track_c", "finetuned_rag"),
    )
    sc4 = cmp.load_item_scores(e4 / "per_item" / "item_scores.jsonl")
    sc4b = cmp.load_item_scores(e4b / "per_item" / "item_scores.jsonl")
    topic_of: dict[str, str] = {}
    with (ROOT / "data/manifests/split_manifest.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            m = json.loads(line)
            topic_of[m["record_id"]] = m["topic"] or ""
    # Copy rate needs the evidence CHUNKS supplied in each prompt; re-derive them with the frozen retriever.
    from medquad_qa.retrieval.factory import build_retriever
    from medquad_qa.retrieval.settings import RetrievalSettings

    bundle = build_retriever(RetrievalSettings.from_env())
    retriever: Any = bundle.retriever
    if retriever is None or getattr(retriever, "name", None) != EXPECTED_RETRIEVER:
        print(f"retriever unavailable or not {EXPECTED_RETRIEVER}: {bundle.statuses}", file=sys.stderr)
        return 3

    def retrieve(q: str) -> tuple[list[tuple[str, str]], list[str]]:
        if hasattr(retriever, "retrieve_with_info"):
            hits, _name, warnings = retriever.retrieve_with_info(q, TOP_K)
        else:
            hits, warnings = retriever.retrieve(q, TOP_K), []
        return [(h.record_id, h.evidence_text) for h in hits], list(warnings)

    runs_a = {"rag": rag_a, "finetuned_rag_v1": v1_a, "finetuned_rag_v2c": v2_a}
    supplied: dict[str, dict[str, list[str]]] = {}
    mismatches: dict[str, int] = {}
    for label, rs in runs_a.items():
        supplied[label], mismatches[label] = cmp.supplied_chunk_texts(rs, a_ex, retrieve)
    sft_rows = [json.loads(line) for line in Path(args.sft_manifest).read_text(encoding="utf-8").splitlines() if line]
    roles = cmp.answer_trained_probe_ids(sft_rows, list(p_ex.values()))

    out = {
        "provenance": {
            "e4_metrics_sha256": sha256_file(e4 / "metrics.json"),
            "e4b_metrics_sha256": sha256_file(e4b / "metrics.json"),
            "shared_scoring_provenance": {k: m4["provenance"][k] for k in cmp.PROVENANCE_KEYS} | {"nli": m4["nli"]},
            "e4b_compare_py_sha256": sha256_file(Path(cmp.__file__)),
            "compare_e4b_py_sha256": sha256_file(Path(__file__)),
            "sft_manifest_sha256": sha256_file(args.sft_manifest),
            "e4b_finetuned_model_version": next(iter(v2_a.values())).model_version if v2_a else None,
        },
        "primary": cmp.primary_family(
            a_ex,
            c_ex,
            rag_a,
            v1_a,
            v2_a,
            rag_c,
            v2_c,
            sc4.get(("test_track_a", "rag"), {}),
            sc4b.get(("test_track_a", "finetuned_rag"), {}),
        ),
        "track_c_descriptive": cmp.track_c_descriptive(
            c_ex, {"rag": rag_c, "finetuned_rag_v1": v1_c, "finetuned_rag_v2c": v2_c}
        ),
        "regression_check": cmp.regression_check(
            a_ex, sc4.get(("test_track_a", "finetuned"), {}), sc4b.get(("test_track_a", "finetuned"), {})
        ),
        "probe_secondary": {
            "answer_trained_roles": dict(sorted(Counter(roles.values()).items())),
            **cmp.probe_secondary(
                p_ex, roles, sc4.get(("train_probe", "finetuned"), {}), sc4b.get(("train_probe", "finetuned"), {})
            ),
        },
        "risk_metrics": cmp.risk_metrics(a_ex, runs_a, supplied, mismatches),
        "amendment_E_descriptive": cmp.descriptive_topic_analyses(
            a_ex,
            c_ex,
            topic_of,
            {"rag": rag_a, "finetuned_rag_v1": v1_a, "finetuned_rag_v2c": v2_a},
            {"rag": rag_c, "finetuned_rag_v1": v1_c, "finetuned_rag_v2c": v2_c},
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print("family size", out["primary"]["family_size"], "->", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
