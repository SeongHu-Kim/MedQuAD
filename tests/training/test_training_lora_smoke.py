"""End-to-end LoRA smoke on a tiny random Qwen3 (CPU): train a few steps, write manifests, reload via
load_generator('finetuned') and generate. Synthetic data only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from training_fixtures import make_split, write_jsonl

from medquad_qa.contracts.interfaces import ChatMessage, ContractViolationError
from medquad_qa.contracts.qa import GenerationParams
from medquad_qa.models import GeneratorConfig, ModelSettings, load_generator
from medquad_qa.training.lora import SFTTrainConfig, check_split_isolation, run_sft


def _tiny_cfg() -> SFTTrainConfig:
    return SFTTrainConfig(
        max_seq_len=512,
        per_device_batch_size=2,
        grad_accum_steps=1,
        bf16=False,
        gradient_checkpointing=True,
        logging_steps=1,
        save_steps=1000,
        eval_max_examples=4,
    )


def test_split_isolation_detects_shared_group(exports: dict[str, Path], tmp_path: Path) -> None:
    assert check_split_isolation(exports["train"], [exports["val"], exports["test"]])["overlap_groups"] == 0
    leaky = make_split("train", 500)  # same group ids as train
    bad = write_jsonl(leaky, tmp_path / "leaky.jsonl")
    with pytest.raises(ContractViolationError):
        check_split_isolation(exports["train"], [bad])


def test_lora_smoke_train_and_reload(exports: dict[str, Path], tiny_model_dir: Path, tmp_path: Path) -> None:
    base = GeneratorConfig(model_id=str(tiny_model_dir), revision="local")
    out_root = tmp_path / "models"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(tiny_model_dir)
    tok = AutoTokenizer.from_pretrained(tiny_model_dir)
    manifest = run_sft(
        _tiny_cfg(),
        base=base,
        train_path=exports["train"],
        val_path=exports["val"],
        heldout_paths=[exports["val"], exports["test"]],
        out_root=out_root,
        run_kind="smoke",
        max_steps=3,
        device="cpu",
        model=model,
        tokenizer=tok,
    )
    t = manifest["training"]
    assert manifest["status"] == "smoke" and t["steps_done"] == 3 == t["planned_steps"]
    assert manifest["data"]["split_isolation"]["overlap_groups"] == 0
    assert manifest["data"]["sft_build_train"]["n_examples"] == 8
    run_dir = out_root / "runs" / manifest["run_id"]
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert any("loss" in r for r in rows) and sum("eval_loss" in r for r in rows) == 2
    assert (run_dir / "run_manifest.json").is_file()
    assert manifest["tracking"]["mlflow_run_id"] and manifest["tracking"]["mlflow_error"] is None

    settings = ModelSettings(
        generator=base, device="cpu", model_dir=out_root, adapter_dir=Path(manifest["adapter"]["dir"])
    )
    adapter_files = [f for f in Path(manifest["adapter"]["dir"]).iterdir() if f.is_file()]
    assert adapter_files and all(f.stat().st_mode & 0o004 for f in adapter_files)  # readable by the API container uid
    gen = load_generator("finetuned", settings)
    assert gen.model_version.endswith(f"+lora:{manifest['run_id']}")
    out = gen.generate([ChatMessage(role="user", content="How is Gout treated?")], GenerationParams(max_new_tokens=4))
    assert out.completion_tokens > 0
