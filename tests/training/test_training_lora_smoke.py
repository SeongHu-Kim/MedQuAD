"""End-to-end LoRA smoke on a tiny random Qwen3 (CPU): train a few steps, write manifests, reload via
load_generator('finetuned') and generate. Synthetic data only."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from training_fixtures import make_split, write_jsonl

from medquad_qa.contracts.interfaces import ChatMessage, ContractViolationError
from medquad_qa.contracts.qa import GenerationParams
from medquad_qa.models import GeneratorConfig, ModelSettings, load_generator
from medquad_qa.training.lora import SFTTrainConfig, check_split_isolation, run_sft
from medquad_qa.training.sft_data import default_closed_book_messages as default_messages


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


def test_lora_smoke_train_and_reload(
    exports: dict[str, Path], tiny_model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # exercise the offline SQLite fallback (written under tmp_path) regardless of the caller's environment,
    # e.g. MEDQUAD_MLFLOW_TRACKING_URI=off in CI
    monkeypatch.delenv("MEDQUAD_MLFLOW_TRACKING_URI", raising=False)
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
    assert manifest["data"]["sft_build"]["train"]["n_examples"] == 8
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


def test_mixed_v2_smoke_two_eval_sets(tiny_model_dir: Path, tmp_path: Path) -> None:
    from test_training_answerability import _records
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from medquad_qa.retrieval.embedding import HashingEmbedder
    from medquad_qa.training.sft_rag_data import MixPlan, mixed_data

    train = write_jsonl(_records("tr", 10, 0), tmp_path / "records_train.jsonl")
    val = write_jsonl(_records("va", 6, 1000), tmp_path / "records_validation.jsonl")
    factory = mixed_data(
        train,
        val,
        HashingEmbedder(),
        MixPlan(n_rag_answerable=4, n_rag_insufficient=2, n_closed_book=4, k=3),
        MixPlan(n_rag_answerable=2, n_rag_insufficient=1, n_closed_book=0, k=3),
        max_seq_len=4096,
        closed_book_builder=default_messages,
        closed_val_max=4,
    )
    manifest = run_sft(
        SFTTrainConfig(max_seq_len=4096, per_device_batch_size=2, grad_accum_steps=1, bf16=False, logging_steps=1),
        base=GeneratorConfig(model_id=str(tiny_model_dir), revision="local"),
        train_path=train,
        val_path=val,
        heldout_paths=[val],
        out_root=tmp_path / "models",
        run_kind="smoke",
        max_steps=2,
        device="cpu",
        model=AutoModelForCausalLM.from_pretrained(tiny_model_dir),
        tokenizer=AutoTokenizer.from_pretrained(tiny_model_dir),
        data_versions={"split_version": "split-x", "mix_version": "sft-mix-v2"},
        data_factory=factory,
    )
    by_set = manifest["training"]["val_loss_by_set"]
    assert set(by_set) == {"eval", "eval_rag"} and by_set["eval_rag"]["examples"] == 3
    assert manifest["data"]["sft_build"]["train"]["counts"] == {
        "rag_answerable": 4,
        "rag_insufficient": 2,
        "closed_book": 4,
    }
    rows = [json.loads(x) for x in (tmp_path / "models" / "runs" / manifest["run_id"] / "sft_record_ids.jsonl").open()]
    assert {r["format"] for r in rows} == {"rag_answerable", "rag_insufficient", "closed_book"}
    assert all(r["split_version"] == "split-x" for r in rows)
    assert manifest["adapter"]["manifest"]["mix_version"] == "sft-mix-v2"
    val_rows = (tmp_path / "models" / "runs" / manifest["run_id"] / "val_record_ids.jsonl").read_text().splitlines()
    parsed = [json.loads(x) for x in val_rows]
    assert [r["format"] for r in parsed].count("closed_book") == 4
    assert len(parsed) == 7 and all(r["evidence_record_ids"] for r in parsed if r["format"] != "closed_book")


def _load_train_lora_script() -> Any:
    path = Path(__file__).resolve().parents[2] / "scripts" / "training" / "train_lora.py"
    spec = importlib.util.spec_from_file_location("train_lora_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "argv", [["train_lora.py", "--build-only"], ["train_lora.py", "--run-kind", "smoke", "--build-only"]]
)
def test_build_only_without_mix_v2_exits_before_any_work(
    argv: list[str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import transformers

    import medquad_qa.models
    import medquad_qa.models.factory
    import medquad_qa.training.lora

    script = _load_train_lora_script()

    def must_not_run(name: str) -> Any:
        def fail(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError(f"{name} was reached; the --build-only guard must exit first")

        return fail

    for target, attr in [
        (script, "run_sft"),
        (script, "_data_versions"),
        (script, "_mix_v2"),
        (script, "_build_only"),
        (medquad_qa.training.lora, "run_sft"),
        (medquad_qa.training.lora, "load_base_for_training"),
        (medquad_qa.models, "load_generator"),
        (medquad_qa.models.factory, "load_generator"),
    ]:
        monkeypatch.setattr(target, attr, must_not_run(f"{getattr(target, '__name__', target)}.{attr}"))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", must_not_run("AutoTokenizer.from_pretrained"))
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", must_not_run("AutoModelForCausalLM.from_pretrained")
    )
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc:
        script.main()
    assert exc.value.code == 2
    assert "--build-only requires --mix v2" in capsys.readouterr().err  # the guard, not some other argparse error


def test_build_only_with_mix_v2_passes_the_guard() -> None:
    script = _load_train_lora_script()
    args = script.parse_args(["--mix", "v2", "--build-only"])
    assert args.build_only and args.mix == "v2" and args.run_kind is None
    with pytest.raises(SystemExit) as exc:  # training still needs --run-kind
        script.parse_args(["--mix", "v2"])
    assert exc.value.code == 2
