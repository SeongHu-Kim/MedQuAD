"""Fresh-process adapter reload + real generation check; optionally promote the run to adapters/CURRENT.

  .venv/bin/python scripts/training/verify_adapter.py --run-id <run_id> [--promote]

Loads base and fine-tuned views through load_generator (shared weights), generates greedy answers for fixed
generic questions (written for this check, not dataset text), and records both outputs, the adapter weight
sha256 and pass/fail checks in artifacts/models/runs/<run_id>/reload_check.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime

from medquad_qa.contracts.interfaces import ChatMessage
from medquad_qa.contracts.qa import GenerationParams
from medquad_qa.models.factory import load_generator
from medquad_qa.models.settings import ModelSettings
from medquad_qa.training.sft_data import default_closed_book_messages

QUESTIONS = [
    "What is high blood pressure?",
    "What are the symptoms of iron deficiency anemia?",
    "How is asthma treated?",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--promote", action="store_true", help="write adapters/CURRENT if all checks pass")
    ap.add_argument("--prompt-builder", choices=["default", "pipeline"], default="pipeline")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    settings = ModelSettings.from_env()
    adapter_dir = settings.model_dir / "adapters" / args.run_id
    run_dir = settings.model_dir / "runs" / args.run_id
    settings = settings.model_copy(update={"adapter_dir": adapter_dir})
    if args.prompt_builder == "pipeline":
        from medquad_qa.rag.prompts import build_closed_book_messages as build
    else:
        build = default_closed_book_messages

    base = load_generator("base", settings)
    tuned = load_generator("finetuned", settings)
    params = GenerationParams(max_new_tokens=args.max_new_tokens)
    rows = []
    for q in QUESTIONS:
        msgs: list[ChatMessage] = build(q)
        b, t = base.generate(msgs, params), tuned.generate(msgs, params)
        rows.append(
            {
                "question": q,
                "base": b.model_dump(),
                "finetuned": t.model_dump(),
                "outputs_differ": b.text != t.text,
            }
        )
    weights = adapter_dir / "adapter_model.safetensors"
    checks = {
        "finetuned_version_has_run_id": tuned.model_version.endswith(f"+lora:{args.run_id}"),
        "shared_backend": tuned.backend is base.backend,
        "all_nonempty": all(r["finetuned"]["text"] for r in rows),
        "any_output_differs_from_base": any(r["outputs_differ"] for r in rows),
    }
    report = {
        "run_id": args.run_id,
        "model_version": tuned.model_version,
        "adapter_weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest() if weights.exists() else None,
        "device": str(tuned.backend.device),
        "params": params.model_dump(),
        "checks": checks,
        "passed": all(checks.values()),
        "samples": rows,
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "reload_check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": checks}, indent=2))
    if args.promote:
        if not report["passed"]:
            print("not promoting: checks failed", file=sys.stderr)
            return 1
        (settings.model_dir / "adapters" / "CURRENT").write_text(args.run_id + "\n", encoding="utf-8")
        print(f"promoted {args.run_id} -> {settings.model_dir / 'adapters' / 'CURRENT'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
