"""Closed-book LoRA SFT on the approved TRAIN export (D-013).

Smoke first (sets the projection used to decide max_steps; tell the lead before the main run):
  .venv/bin/python scripts/training/train_lora.py --run-kind smoke --max-steps 20
Main run:
  .venv/bin/python scripts/training/train_lora.py --run-kind main [--max-steps N]
Then verify + promote in a fresh process:
  .venv/bin/python scripts/training/verify_adapter.py --run-id <run_id> --promote
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from medquad_qa.models.settings import ModelSettings
from medquad_qa.training.lora import DEFAULT_TRAIN_CONFIG, SFTTrainConfig, run_sft
from medquad_qa.training.sft_data import MessageBuilder, default_closed_book_messages

EXPORTS = Path("data/processed/exports")


def _message_builder(name: str) -> tuple[MessageBuilder, str]:
    if name == "default":
        return default_closed_book_messages, "training.sft_data.default_closed_book_messages"
    if name == "pipeline":  # the RAG pipeline's frozen closed-book builder (retrieval-engineer)
        from medquad_qa.rag import prompts

        return prompts.build_closed_book_messages, f"rag.prompts.build_closed_book_messages@{prompts.PROMPT_VERSION}"
    raise SystemExit(f"unknown --prompt-builder {name}")


def _data_versions(exports_dir: Path) -> dict[str, Any]:
    manifest = exports_dir.parent.parent / "manifests" / "exports_manifest.json"
    if not manifest.is_file():
        manifest = Path("data/manifests/exports_manifest.json")
    if not manifest.is_file():
        return {"exports_manifest": None}
    data = json.loads(manifest.read_text(encoding="utf-8"))
    keep = ("corpus_version", "split_version", "files")
    return {"exports_manifest": str(manifest), **{k: data[k] for k in keep if k in data}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_TRAIN_CONFIG))
    ap.add_argument("--exports-dir", default=str(EXPORTS))
    ap.add_argument("--run-kind", choices=["smoke", "main"], required=True)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--prompt-builder", choices=["default", "pipeline"], default="pipeline")
    ap.add_argument("--out-root", default=None, help="defaults to MEDQUAD_MODEL_DIR (artifacts/models)")
    args = ap.parse_args()

    exports = Path(args.exports_dir)
    paths = {s: exports / f"records_{s}.jsonl" for s in ("train", "val", "test")}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        print(f"missing exports (data-steward D5): {missing}", file=sys.stderr)
        return 2
    settings = ModelSettings.from_env()
    builder, builder_name = _message_builder(args.prompt_builder)
    config = SFTTrainConfig.from_yaml(args.config)
    versions = {**_data_versions(exports), "message_builder": builder_name}

    manifest = run_sft(
        config,
        base=settings.generator,
        train_path=paths["train"],
        val_path=paths["val"],
        heldout_paths=[paths["val"], paths["test"]],
        out_root=Path(args.out_root) if args.out_root else settings.model_dir,
        run_kind=args.run_kind,
        max_steps=args.max_steps,
        device=args.device,
        message_builder=builder,
        data_versions=versions,
    )
    t = manifest["training"]
    summary = {
        "run_id": manifest["run_id"],
        "status": manifest["status"],
        "steps_done": t["steps_done"],
        "steps_per_epoch": t["steps_per_epoch"],
        "sec_per_step_mean": t["sec_per_step_mean"],
        "projected_full_epoch_h": (t["sec_per_step_mean"] or 0) * t["steps_per_epoch"] / 3600,
        "val_loss_before": t["val_loss_before"],
        "val_loss_after": t["val_loss_after"],
        "cuda_max_allocated_gib": t["cuda_max_allocated_gib"],
        "n_truncated": manifest["data"]["sft_build_train"]["n_truncated"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
