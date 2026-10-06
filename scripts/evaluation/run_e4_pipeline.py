"""E4: run the frozen QA pipeline over frozen eval sets in the requested experiment modes.

Usage (TEST runs only after the E4 leakage gate passes and configs are frozen, D-035):
  HF_HUB_OFFLINE=1 TORCH_DISABLE_NATIVE_JIT=1 MEDQUAD_RETRIEVER=dense_fallback \
  MEDQUAD_INDEX_TEXT_MODE=question_answer MEDQUAD_GATE_MODE=auto \
  python scripts/evaluation/run_e4_pipeline.py --evalset artifacts/evaluation/evalsets/test_track_a.jsonl \
      --modes base rag finetuned finetuned_rag --out-dir artifacts/evaluation/e4/runs \
      --expect safety_rules_version=safety-v2+43836b6c --expect prompt_version=rag-v1+df593554

Per-item QAResponses go to <out-dir>/per_item/<set>__<mode>.jsonl (gitignored: answer text). A run manifest with
versions, git sha, env and timing goes to <out-dir>/<set>__<mode>.meta.json. Items with ``injected_evidence`` are
answered by a pipeline sharing the same generators and gate but a FixtureRetriever (RAG modes only). Closed-book
modes skip fixture items. Resumable: items already present in the per-item file are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from medquad_qa.contracts import RAG_MODES, MedQuADError, QARequest
from medquad_qa.evaluation.evalset import load_eval_set, sha256_file
from medquad_qa.evaluation.fixture_retriever import FixtureRetriever

ROOT = Path(__file__).resolve().parents[2]
TOP_K = 5
ENV_KEYS = ("MEDQUAD_RETRIEVER", "MEDQUAD_INDEX_TEXT_MODE", "MEDQUAD_GATE_MODE", "MEDQUAD_QDRANT_URL", "HF_HUB_OFFLINE")


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)  # noqa: S603,S607
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def read_done(path: Path) -> dict[str, dict[str, Any]]:
    """Rows already written (resume). A trailing partial line from a killed run is truncated away."""
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    data = path.read_bytes()
    good_end = 0
    for line in data.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            break
        if row["example_id"] in rows:
            raise ValueError(f"{path}: duplicated example_id {row['example_id']}")
        rows[row["example_id"]] = row
        good_end += len(line)
    if good_end != len(data):
        with path.open("r+b") as fh:
            fh.truncate(good_end)
        print(f"{path.name}: truncated {len(data) - good_end} bytes of partial output", file=sys.stderr)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evalset", required=True)
    ap.add_argument("--modes", nargs="+", required=True)
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts/evaluation/e4/runs"))
    ap.add_argument("--expect", action="append", default=[], help="key=value that pipeline.versions() must match")
    ap.add_argument("--limit", type=int, default=None, help="smoke runs only")
    ap.add_argument(
        "--notes-prefix",
        nargs="*",
        default=None,
        help="pre-declared subset: keep only items whose notes start with one of these tags",
    )
    ap.add_argument("--allow-fallback", action="store_true", help="permit a degraded dense retriever (not for E4)")
    ap.add_argument("--smoke-fake-generator", action="store_true", help="plumbing check without model weights")
    args = ap.parse_args()

    from medquad_qa.rag.factory import build_pipeline
    from medquad_qa.rag.pipeline import RagPipeline

    examples = load_eval_set(args.evalset)
    if args.notes_prefix:
        examples = [e for e in examples if (e.notes or "").startswith(tuple(args.notes_prefix))]
    if args.limit:
        examples = examples[: args.limit]
    set_name = Path(args.evalset).stem
    t0 = time.perf_counter()
    if args.smoke_fake_generator:
        from medquad_qa.evaluation.fakes import ScriptedGenerator

        fake = ScriptedGenerator("Synthetic smoke answer [E1].")
        pipe = build_pipeline(generator_provider=lambda _v: fake)
    else:
        pipe = build_pipeline()
    build_s = time.perf_counter() - t0
    versions = pipe.versions()
    degraded = [c.name for c in pipe.readiness() if c.name.startswith("retriever:") and not c.ok]
    if degraded and not args.allow_fallback:
        # e.g. embedded Qdrant locked by another process -> silent-but-flagged BM25 fallback; never measure that as E4
        print(
            f"retriever components unavailable: {degraded}; refusing (use --allow-fallback only for smoke)",
            file=sys.stderr,
        )
        return 3
    for kv in args.expect:
        k, v = kv.split("=", 1)
        if versions.get(k) != v:
            print(f"version mismatch: {k}={versions.get(k)!r}, expected {v!r}", file=sys.stderr)
            return 2

    out = Path(args.out_dir)
    (out / "per_item").mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        rag = mode in RAG_MODES
        path = out / "per_item" / f"{set_name}__{mode}.jsonl"
        done = set(read_done(path))
        n_resumed = len(done)
        n_err = n_skip = n_new = 0
        started = datetime.now(UTC).isoformat(timespec="seconds")
        tm = time.perf_counter()
        with path.open("a", encoding="utf-8") as fh:
            for e in examples:
                if e.example_id in done:
                    continue
                if e.injected_evidence is not None and not rag:
                    n_skip += 1
                    continue
                runner: Any = pipe
                if e.injected_evidence is not None:
                    runner = RagPipeline(
                        retriever=FixtureRetriever({e.question: e.injected_evidence}),
                        generator_provider=pipe.get_generator,
                        gate=pipe.gate,
                    )
                rid = f"e4-{set_name}-{mode}-{e.example_id}"
                row: dict[str, Any] = {
                    "example_id": e.example_id,
                    "mode": mode,
                    "eval_split": e.eval_split,
                    "case_type": e.case_type,
                    "label_provenance": e.label_provenance,
                    "question_provenance": e.question_provenance,
                    "reviewer": e.reviewer,
                }
                ti = time.perf_counter()
                try:
                    resp = runner.answer(QARequest(question=e.question, experiment_mode=mode, top_k=TOP_K), rid)
                    row["response"] = resp.model_dump(mode="json")
                except MedQuADError as exc:
                    n_err += 1
                    row["error"] = type(exc).__name__
                row["item_wall_ms"] = round((time.perf_counter() - ti) * 1000, 1)
                n_new += 1
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
        meta = {
            "set": set_name,
            "evalset_sha256": sha256_file(args.evalset),
            "mode": mode,
            "n_examples": len(examples),
            "n_errors_this_session": n_err,
            "n_resumed_rows": n_resumed,
            "n_new_rows_this_session": n_new,
            "n_rows_total": len(read_done(path)),
            "evalset_manifest_sha256": sha256_file(f"{args.evalset}.manifest.json"),
            "n_skipped_fixture_items": n_skip,
            "top_k": TOP_K,
            "versions": versions,
            "readiness": [c.model_dump() for c in pipe.readiness()],
            "git_sha": git_sha(),
            "env": {k: os.environ.get(k) for k in ENV_KEYS},
            "platform": platform.platform(),
            "build_pipeline_s": round(build_s, 1),
            "started_at": started,
            "wall_s": round(time.perf_counter() - tm, 1),
            "limit": args.limit,
            "notes_prefix": args.notes_prefix,
            "smoke_fake_generator": args.smoke_fake_generator,
            "per_item_file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        }
        (out / f"{set_name}__{mode}.meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
        print(mode, "done", meta["wall_s"], "s errors", n_err, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
