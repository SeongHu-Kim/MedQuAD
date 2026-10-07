"""Closed-book LoRA SFT on the approved TRAIN export (D-013).

Smoke first (sets the projection used to decide max_steps; tell the lead before the main run):
  .venv/bin/python scripts/training/train_lora.py --run-kind smoke --max-steps 20
Main run:
  .venv/bin/python scripts/training/train_lora.py --run-kind main [--max-steps N]
Then verify + promote in a fresh process:
  .venv/bin/python scripts/training/verify_adapter.py --run-id <run_id> --promote
Adapter v2 (mixed closed-book + RAG-cited, D-051):
  .venv/bin/python scripts/training/train_lora.py --mix v2 --config configs/training/lora_sft_mixed_v2.yaml \
      --run-kind smoke --max-steps 20 --expect-prompt-version cb-v1+a1f08aaf --expect-rag-prompt-version rag-v1+df593554
  .venv/bin/python scripts/training/train_lora.py --mix v2 ... --build-only     # CPU: build + IDs manifest, no training
  sft-mix-v2b (option b'): the same commands with --mix-config configs/training/mix_v2b.yaml
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

        return prompts.build_closed_book_messages, prompts.CLOSED_BOOK_PROMPT_VERSION
    raise SystemExit(f"unknown --prompt-builder {name}")


def _data_versions() -> dict[str, Any]:
    """Verify every export's sha256 against data/manifests/exports_manifest.json (data-steward's loader)."""
    from medquad_qa.data.corpus import load_split

    loaded = {s: load_split(s) for s in ("train", "validation", "test")}  # raises ContractViolationError on drift
    train = loaded["train"]
    return {
        "exports_manifest": "data/manifests/exports_manifest.json",
        "corpus_version": train.corpus_version,
        "split_version": train.split_version,
        "export_sha256_verified": {s: v.sha256 for s, v in loaded.items()},
    }


def _mix_v2(
    args: Any, paths: dict[str, Path], closed_builder: MessageBuilder, config: Any
) -> tuple[Any, dict[str, Any]]:
    import yaml

    from medquad_qa.rag import prompts
    from medquad_qa.retrieval.embedding import SentenceTransformerEmbedder
    from medquad_qa.retrieval.settings import DEFAULT_QUERY_PREFIX
    from medquad_qa.training.sft_rag_data import MIX_VERSION, MixPlan, mixed_data

    if args.expect_rag_prompt_version and args.expect_rag_prompt_version != prompts.PROMPT_VERSION:
        print(f"RAG prompt {prompts.PROMPT_VERSION!r} != {args.expect_rag_prompt_version!r}; aborting", file=sys.stderr)
        return None, {}
    mix = yaml.safe_load(Path(args.mix_config).read_text(encoding="utf-8"))
    emb_cfg = mix["embedder"]
    embedder = SentenceTransformerEmbedder(
        emb_cfg["model_id"], emb_cfg["revision"], query_prefix=DEFAULT_QUERY_PREFIX, device=emb_cfg["device"]
    )
    # mix_version and the optional v2b block apply to both the train and the validation-RAG plan
    shared = {"mix_version": mix.get("mix_version", MIX_VERSION), **mix.get("v2b", {})}
    factory = mixed_data(
        paths["train"],
        paths["validation"],
        embedder,
        MixPlan(**shared, **mix["train"]),
        MixPlan(**shared, **mix["validation_rag"]),
        max_seq_len=config.max_seq_len,
        closed_book_builder=closed_builder,
        closed_val_max=mix["closed_val_max"],
    )
    return factory, {
        "mix_version": shared["mix_version"],
        "mix_config": mix,
        "rag_prompt_version": prompts.PROMPT_VERSION,
        "embedder": {
            "model_id": embedder.model_id,
            "revision": embedder.revision,
            "query_prefix": DEFAULT_QUERY_PREFIX,
        },
    }


def _build_only(factory: Any, settings: ModelSettings, versions: dict[str, Any], out_root: Path) -> int:
    """CPU-only: tokenize + build the mix, write reports and the IDs-only manifest for the leakage gate."""
    from datetime import UTC, datetime

    from transformers import AutoTokenizer

    cfg = settings.generator
    tok = AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.revision, local_files_only=True)
    data = factory(tok)
    out = out_root / "builds" / f"{versions['mix_version']}-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    for name, rep in data.reports.items():
        (out / f"sft_build_{name}.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    with (out / "sft_record_ids.jsonl").open("w", encoding="utf-8") as f:
        for row in data.id_rows:
            f.write(json.dumps({**row, "split_version": versions.get("split_version")}) + "\n")
    with (out / "val_record_ids.jsonl").open("w", encoding="utf-8") as f:  # pass as --threshold-ids to the gate
        for row in data.val_id_rows:
            f.write(json.dumps({**row, "split_version": versions.get("split_version")}) + "\n")
    tokens = sum(len(e.input_ids) for e in data.train)
    summary = {
        "out": str(out),
        "train_examples": len(data.train),
        "train_tokens": tokens,
        "val_sets": {k: len(v) for k, v in data.val_sets.items()},
        "versions": {k: v for k, v in versions.items() if k != "mix_config"},
    }
    (out / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate arguments. Invalid combinations exit with code 2 before any data or model work."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_TRAIN_CONFIG))
    ap.add_argument("--exports-dir", default=str(EXPORTS))
    ap.add_argument("--run-kind", choices=["smoke", "main"], default=None, help="required unless --build-only")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--prompt-builder", choices=["default", "pipeline"], default="pipeline")
    ap.add_argument("--out-root", default=None, help="defaults to MEDQUAD_MODEL_DIR (artifacts/models)")
    ap.add_argument("--expect-prompt-version", default=None, help="abort unless the builder version equals this")
    ap.add_argument("--mix", choices=["none", "v2"], default="none")
    ap.add_argument("--mix-config", default="configs/training/mix_v2.yaml")
    ap.add_argument("--expect-rag-prompt-version", default=None)
    ap.add_argument("--build-only", action="store_true", help="build examples + manifests only (no model, no GPU)")
    args = ap.parse_args(argv)
    if args.build_only and args.mix != "v2":
        # without this guard --build-only fell through to run_sft (model load + training)
        ap.error("--build-only requires --mix v2 (it builds the mixed v2 data and manifests; it never trains)")
    if not args.build_only and args.run_kind is None:
        ap.error("--run-kind is required unless --build-only")
    return args


def main() -> int:
    args = parse_args()

    exports = Path(args.exports_dir)
    paths = {s: exports / f"records_{s}.jsonl" for s in ("train", "validation", "test")}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        print(f"missing exports (data-steward D5): {missing}", file=sys.stderr)
        return 2
    settings = ModelSettings.from_env()
    builder, builder_name = _message_builder(args.prompt_builder)
    if args.expect_prompt_version and builder_name != args.expect_prompt_version:
        print(f"prompt version {builder_name!r} != expected {args.expect_prompt_version!r}; aborting", file=sys.stderr)
        return 3
    config = SFTTrainConfig.from_yaml(args.config)
    versions = {**_data_versions(), "message_builder": builder_name}
    factory = None
    if args.mix == "v2":
        factory, mix_versions = _mix_v2(args, paths, builder, config)
        if factory is None:
            return 5
        versions.update(mix_versions)
        if args.build_only:
            return _build_only(
                factory, settings, versions, Path(args.out_root) if args.out_root else settings.model_dir
            )

    manifest = run_sft(
        config,
        base=settings.generator,
        train_path=paths["train"],
        val_path=paths["validation"],
        heldout_paths=[paths["validation"], paths["test"]],
        out_root=Path(args.out_root) if args.out_root else settings.model_dir,
        run_kind=args.run_kind,
        max_steps=args.max_steps,
        device=args.device,
        message_builder=builder,
        data_versions=versions,
        data_factory=factory,
    )
    recorded = manifest["adapter"]["manifest"]["message_builder"]
    if args.expect_prompt_version and recorded != args.expect_prompt_version:
        print(f"adapter manifest records {recorded!r}, expected {args.expect_prompt_version!r}", file=sys.stderr)
        return 4
    recorded_rag = manifest["adapter"]["manifest"].get("rag_prompt_version")
    if args.expect_rag_prompt_version and recorded_rag != args.expect_rag_prompt_version:
        print(
            f"adapter manifest records RAG {recorded_rag!r}, expected {args.expect_rag_prompt_version!r}",
            file=sys.stderr,
        )
        return 4
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
        "val_loss_by_set": t["val_loss_by_set"],
        "cuda_max_allocated_gib": t["cuda_max_allocated_gib"],
        "train_build": manifest["data"]["sft_build"]["train"].get(
            "n_truncated", manifest["data"]["sft_build"]["train"].get("counts")
        ),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
