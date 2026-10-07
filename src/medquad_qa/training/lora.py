"""Prompt-masked LoRA SFT with Hugging Face Trainer + PEFT, with run manifests and MLflow logging.

Output layout (``out_root`` = artifacts/models):
  adapters/<run_id>/   adapter weights + adapter_config.json + adapter_manifest.json (loadable by load_generator)
  runs/<run_id>/       run_manifest.json, metrics.jsonl, sft_build_*.json, loss_curve.png, trainer/ (checkpoints)
``adapters/CURRENT`` is NOT updated here; scripts/training/verify_adapter.py promotes a run after a fresh-process
reload + real generation check.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field

from medquad_qa.contracts.interfaces import ContractViolationError
from medquad_qa.models.factory import ADAPTER_MANIFEST
from medquad_qa.models.settings import GeneratorConfig
from medquad_qa.models.torch_runtime import apply_native_jit_guard
from medquad_qa.training.sft_data import (
    IGNORE_INDEX,
    MessageBuilder,
    SFTBuilder,
    SFTExample,
    build_from_file,
    default_closed_book_messages,
    file_sha256,
    read_records,
)

DEFAULT_TRAIN_CONFIG = Path("configs/training/lora_sft.yaml")
EXPERIMENT = "medquad-sft"
PROBE_QUESTION = "What is (are) example condition?"


class LoraParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str | list[str] = "all-linear"


class OptimParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_fraction: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0


class SFTTrainConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    format_version: str = "sft-closed-v1"
    max_seq_len: int = 1024
    seed: int = 42
    lora: LoraParams = Field(default_factory=LoraParams)
    optim: OptimParams = Field(default_factory=OptimParams)
    per_device_batch_size: int = 4
    grad_accum_steps: int = 4
    num_train_epochs: float = 1.0
    max_steps: int | None = None
    time_cap_hours: float = 2.5
    bf16: bool = True
    gradient_checkpointing: bool = True
    sampling: str = "group_by_length"
    logging_steps: int = 10
    save_steps: int = 200
    eval_max_examples: int = 512

    @classmethod
    def from_yaml(cls, path: str | Path) -> SFTTrainConfig:
        return cls(**(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}))

    def fingerprint(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:8]


# --------------------------------------------------------------------------- data plumbing
class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, examples: list[SFTExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> dict[str, Any]:
        ex = self.examples[i]
        return {"input_ids": ex.input_ids, "labels": ex.labels, "length": len(ex.input_ids)}


class PadCollator:
    """Right-pads input_ids with pad_id, labels with IGNORE_INDEX; builds attention_mask."""

    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        width = max(len(b["input_ids"]) for b in batch)
        ids = [b["input_ids"] + [self.pad_id] * (width - len(b["input_ids"])) for b in batch]
        labels = [b["labels"] + [IGNORE_INDEX] * (width - len(b["labels"])) for b in batch]
        mask = [[1] * len(b["input_ids"]) + [0] * (width - len(b["input_ids"])) for b in batch]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


def check_split_isolation(train_path: Path, other_paths: list[Path]) -> dict[str, int]:
    """Raise ContractViolationError if any record or split group of TRAIN appears in val/test exports."""
    train = list(read_records(train_path))
    groups = {r.split_group_id for r in train}
    ids = {r.record_id for r in train}
    overlap_groups = overlap_ids = 0
    for p in other_paths:
        for r in read_records(p):
            overlap_groups += r.split_group_id in groups
            overlap_ids += r.record_id in ids
    if overlap_groups or overlap_ids:
        raise ContractViolationError(
            f"train overlaps held-out exports: {overlap_groups} records share a split group, {overlap_ids} share an id"
        )
    return {"train_records": len(train), "train_groups": len(groups), "overlap_groups": 0, "overlap_ids": 0}


# --------------------------------------------------------------------------- tracking
class _Tracker:
    """MLflow if reachable, else the local file store; never fails the run (metrics.jsonl is primary)."""

    def __init__(self, run_name: str, metrics_path: Path, fallback_dir: Path) -> None:
        self.metrics_path = metrics_path
        self.mlflow: Any = None
        self.tracking_uri: str | None = None
        self.mlflow_run_id: str | None = None
        self.error: str | None = None
        uri = os.environ.get("MEDQUAD_MLFLOW_TRACKING_URI", "")
        if uri == "off":
            self.error = "disabled (MEDQUAD_MLFLOW_TRACKING_URI=off)"
            return
        try:
            import mlflow

            artifact_location = None
            if not (uri.startswith("http") and _reachable(uri)):
                # MLflow 3.16 refuses the plain file store, so the offline fallback is SQLite in the same directory
                fallback_dir.mkdir(parents=True, exist_ok=True)
                uri = f"sqlite:///{fallback_dir.resolve() / 'mlflow.db'}"
                artifact_location = (fallback_dir.resolve() / "artifacts").as_uri()
            mlflow.set_tracking_uri(uri)
            if mlflow.get_experiment_by_name(EXPERIMENT) is None:
                mlflow.create_experiment(EXPERIMENT, artifact_location=artifact_location)
            mlflow.set_experiment(EXPERIMENT)
            run = mlflow.start_run(run_name=run_name)
            self.mlflow, self.tracking_uri, self.mlflow_run_id = mlflow, uri, run.info.run_id
        except Exception as exc:  # tracking is best effort
            self.error = f"{type(exc).__name__}: {exc}"

    def params(self, params: dict[str, Any]) -> None:
        if self.mlflow:
            self.mlflow.log_params({k: str(v)[:500] for k, v in params.items()})

    def metrics(self, metrics: dict[str, float], step: int) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")
        if self.mlflow:
            self.mlflow.log_metrics(metrics, step=step)

    def artifact(self, path: Path) -> None:
        if self.mlflow and path.exists():
            self.mlflow.log_artifact(str(path))

    def end(self, status: str) -> None:
        if self.mlflow:
            self.mlflow.set_tag("medquad.status", status)
            self.mlflow.end_run()


def _reachable(uri: str) -> bool:
    try:
        with urllib.request.urlopen(uri.rstrip("/") + "/health", timeout=2) as resp:  # noqa: S310
            return bool(resp.status == 200)
    except Exception:
        return False


# --------------------------------------------------------------------------- helpers
def _env() -> dict[str, Any]:
    import peft
    import transformers

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()  # noqa: S603, S607
    except (OSError, subprocess.CalledProcessError):
        sha = None
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "machine": platform.machine(),
        "python": platform.python_version(),
        "git_sha": sha,
        "torch_native_triton_ops": apply_native_jit_guard(),
    }


def _tokenizer_meta(tokenizer: Any, builder: SFTBuilder) -> dict[str, Any]:
    template = getattr(tokenizer, "chat_template", None) or ""
    probe = builder.prompt_ids(PROBE_QUESTION)
    return {
        "class": type(tokenizer).__name__,
        "vocab_size": len(tokenizer),
        "eos_token": tokenizer.eos_token,
        "pad_token": tokenizer.pad_token,
        "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "prompt_fingerprint_sha256": hashlib.sha256(json.dumps(probe).encode()).hexdigest(),
        "end_of_turn_ids": builder.eot_ids,
    }


def _plot_loss(metrics_path: Path, out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    train = [(r["step"], r["loss"]) for r in rows if "loss" in r]
    evals = [(r["step"], r["eval_loss"]) for r in rows if "eval_loss" in r]
    if not train:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(*zip(*train, strict=True), label="train loss (assistant tokens)")
    if evals:
        ax.plot(*zip(*evals, strict=True), "o--", label="validation loss")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def load_base_for_training(base: GeneratorConfig, device: torch.device, bf16: bool) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base.model_id, revision=base.revision, local_files_only=True)
    dtype = torch.bfloat16 if (bf16 and device.type == "cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base.model_id,
        revision=base.revision,
        dtype=dtype,
        attn_implementation=base.attn_implementation,
        local_files_only=True,
        disable_mmap=True,  # see models.factory.get_backend
    )
    return model.to(device), tokenizer  # type: ignore[arg-type]


def _make_world_readable(directory: Path) -> None:
    for path in directory.iterdir():
        if path.is_file():
            path.chmod(0o644)


def _disable_adapter_input_casting(model: Any) -> int:
    from peft.tuners.tuners_utils import BaseTunerLayer

    n = 0
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            module.cast_input_dtype_enabled = False  # type: ignore[attr-defined]
            n += 1
    return n


def _write_sft_ids(rows: list[dict[str, Any]], path: Path, split_version: str | None) -> None:
    """IDs-only list of the records actually trained on (for the evaluator's leakage gate; no text)."""
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({**row, "split_version": split_version}) + "\n")


@dataclass
class PreparedData:
    """Tokenised training data. ``val_sets`` maps an eval prefix (e.g. 'eval', 'eval_rag') to its examples."""

    train: list[SFTExample]
    val_sets: dict[str, list[SFTExample]]
    reports: dict[str, dict[str, Any]]  # name -> JSON-able build report (written as sft_build_<name>.json)
    id_rows: list[dict[str, Any]]
    train_sha256: str | None
    val_id_rows: list[dict[str, Any]] = field(default_factory=list)  # validation-built examples (threshold-ids)


DataFactory = Callable[[Any], PreparedData]


def closed_book_data(
    train_path: Path, val_path: Path, max_seq_len: int, message_builder: MessageBuilder, eval_max: int
) -> DataFactory:
    """The v1 data path: closed-book examples from the train export; closed-book validation loss."""

    def factory(tokenizer: Any) -> PreparedData:
        builder = SFTBuilder(tokenizer, max_seq_len=max_seq_len, message_builder=message_builder)
        train_ex, train_report = build_from_file(builder, train_path)
        val_ex, val_report = build_from_file(builder, val_path)
        rows = [
            {"record_id": e.record_id, "split_group_id": e.split_group_id, "truncated": e.truncated} for e in train_ex
        ]
        return PreparedData(
            train=train_ex,
            val_sets={"eval": val_ex[:eval_max]},  # export order is deterministic
            reports={"train": train_report.to_dict(), "val": val_report.to_dict()},
            id_rows=rows,
            train_sha256=train_report.input_sha256,
        )

    return factory


# --------------------------------------------------------------------------- main entry
def run_sft(
    config: SFTTrainConfig,
    *,
    base: GeneratorConfig,
    train_path: Path,
    val_path: Path,
    heldout_paths: list[Path],
    out_root: Path,
    run_kind: str = "main",
    max_steps: int | None = None,
    device: str = "auto",
    message_builder: MessageBuilder = default_closed_book_messages,
    model: Any = None,
    tokenizer: Any = None,
    data_versions: dict[str, Any] | None = None,
    data_factory: DataFactory | None = None,
) -> dict[str, Any]:
    """Train a LoRA adapter on TRAIN-split SFT examples. Returns the run manifest (also written to disk)."""
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

    apply_native_jit_guard()
    started = time.monotonic()
    created = datetime.now(UTC)
    isolation = check_split_isolation(train_path, heldout_paths)
    run_id = f"sft-{run_kind}-{created:%Y%m%d-%H%M%S}-{config.fingerprint()}"
    run_dir = out_root / "runs" / run_id
    adapter_dir = out_root / "adapters" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)

    dev = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    )
    if model is None or tokenizer is None:
        model, tokenizer = load_base_for_training(base, dev, config.bf16)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    use_bf16 = config.bf16 and dev.type == "cuda"

    builder = SFTBuilder(tokenizer, max_seq_len=config.max_seq_len, message_builder=message_builder)
    factory = data_factory or closed_book_data(
        train_path, val_path, config.max_seq_len, message_builder, config.eval_max_examples
    )
    data = factory(tokenizer)
    train_ex = data.train
    for name, rep_dict in data.reports.items():
        (run_dir / f"sft_build_{name}.json").write_text(json.dumps(rep_dict, indent=2) + "\n", encoding="utf-8")
    _write_sft_ids(data.id_rows, run_dir / "sft_record_ids.jsonl", (data_versions or {}).get("split_version"))
    if data.val_id_rows:
        _write_sft_ids(data.val_id_rows, run_dir / "val_record_ids.jsonl", (data_versions or {}).get("split_version"))

    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    lora_cfg = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_cfg)
    # LoRA weights stay fp32 (optimizer precision) but PEFT would also cast every adapter INPUT to fp32, which made
    # a step ~2.5x slower on GB10 (profiled: 3.85 s vs 1.51 s per 4x512 micro-batch). Under bf16 autocast the
    # adapter matmuls run in bf16 either way, so input casting is disabled for training only.
    n_uncast = _disable_adapter_input_casting(peft_model) if use_bf16 else 0
    trainable, total = peft_model.get_nb_trainable_parameters()

    effective_batch = config.per_device_batch_size * config.grad_accum_steps
    steps_per_epoch = math.ceil(len(train_ex) / effective_batch)
    planned_steps = max_steps or config.max_steps or math.ceil(steps_per_epoch * config.num_train_epochs)
    warmup = max(1, math.ceil(config.optim.warmup_fraction * planned_steps))

    # offline fallback store lives under out_root (artifacts/models/mlruns-local for real runs; D-015)
    tracker = _Tracker(run_id, run_dir / "metrics.jsonl", out_root / "mlruns-local")
    cap_s = config.time_cap_hours * 3600.0

    class _Callback(TrainerCallback):
        hit_cap = False
        step_times: list[float] = []  # noqa: RUF012

        def on_step_end(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
            self.step_times.append(time.monotonic())
            if time.monotonic() - started > cap_s:
                self.hit_cap = True
                control.should_training_stop = True
                control.should_save = True

        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kw: Any) -> None:
            numeric = {k: float(v) for k, v in (logs or {}).items() if isinstance(v, int | float)}
            if numeric:
                tracker.metrics(numeric, state.global_step)

    callback = _Callback()
    targs = TrainingArguments(
        output_dir=str(run_dir / "trainer"),
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.grad_accum_steps,
        learning_rate=config.optim.learning_rate,
        lr_scheduler_type=config.optim.lr_scheduler_type,
        warmup_steps=warmup,
        weight_decay=config.optim.weight_decay,
        max_grad_norm=config.optim.max_grad_norm,
        num_train_epochs=config.num_train_epochs,
        max_steps=planned_steps,
        bf16=use_bf16,
        use_cpu=dev.type == "cpu",
        gradient_checkpointing=False,  # enabled on the model above
        train_sampling_strategy=config.sampling,
        logging_steps=config.logging_steps,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=1,
        eval_strategy="no",
        report_to="none",
        seed=config.seed,
        data_seed=config.seed,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        optim="adamw_torch",
    )
    trainer = Trainer(
        model=peft_model,
        args=targs,
        data_collator=PadCollator(int(tokenizer.pad_token_id)),
        train_dataset=SFTDataset(train_ex),
        eval_dataset=SFTDataset(data.val_sets["eval"]),
        callbacks=[callback],
    )

    params = {
        "run_id": run_id,
        "run_kind": run_kind,
        "base_model": base.base_version,
        **{f"cfg.{k}": v for k, v in config.model_dump().items()},
        "planned_steps": planned_steps,
        "steps_per_epoch": steps_per_epoch,
        "train_examples": len(train_ex),
        "val_examples_for_loss": {k: len(v) for k, v in data.val_sets.items()},
        "trainable_params": trainable,
        "total_params": total,
        "train_sha256": data.train_sha256,
    }
    tracker.params(params)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def _evaluate() -> dict[str, float]:  # each set is logged by the callback as <prefix>_loss
        return {
            name: float(trainer.evaluate(eval_dataset=SFTDataset(ex), metric_key_prefix=name)[f"{name}_loss"])
            for name, ex in data.val_sets.items()
        }

    val_before_all = _evaluate()
    val_before = val_before_all["eval"]
    train_t0 = time.monotonic()
    train_out = trainer.train()
    train_s = time.monotonic() - train_t0
    steps_done = int(trainer.state.global_step)
    val_after_all = _evaluate()
    val_after = val_after_all["eval"]

    peft_model.save_pretrained(str(adapter_dir))
    _make_world_readable(adapter_dir)  # safetensors writes 0600; the API container (uid 10001) must read it
    if steps_done < planned_steps:
        status = "partial_time_cap" if callback.hit_cap else "partial"
    else:
        status = "completed" if run_kind == "main" else run_kind
    adapter_manifest = {
        "run_id": run_id,
        "status": status,
        "base_model_id": base.model_id,
        "base_revision": base.revision,
        "format_version": config.format_version,
        "message_builder": (data_versions or {}).get("message_builder"),
        "rag_prompt_version": (data_versions or {}).get("rag_prompt_version"),
        "mix_version": (data_versions or {}).get("mix_version"),
        "tokenizer": _tokenizer_meta(tokenizer, builder),
        "created_at_utc": created.isoformat(timespec="seconds"),
    }
    (adapter_dir / ADAPTER_MANIFEST).write_text(json.dumps(adapter_manifest, indent=2) + "\n", encoding="utf-8")
    weights = adapter_dir / "adapter_model.safetensors"

    step_times = callback.step_times
    sec_per_step = (step_times[-1] - step_times[0]) / (len(step_times) - 1) if len(step_times) > 1 else None
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "run_kind": run_kind,
        "status": status,
        "base_model": {"model_id": base.model_id, "revision": base.revision},
        "config": config.model_dump(),
        "seeds": {"seed": config.seed, "data_seed": config.seed},
        "data": {
            "train_path": str(train_path),
            "val_path": str(val_path),
            "train_sha256": data.train_sha256,
            "val_sha256": file_sha256(val_path),
            "heldout_sha256": {str(p): file_sha256(p) for p in heldout_paths},
            "split_isolation": isolation,
            "versions": data_versions or {},
            "sft_build": data.reports,
        },
        "training": {
            "device": str(dev),
            "precision": "bf16 autocast, fp32 LoRA weights" if use_bf16 else "fp32",
            "adapter_input_casting_disabled_modules": n_uncast,
            "trainable_params": trainable,
            "total_params": total,
            "effective_batch": effective_batch,
            "steps_per_epoch": steps_per_epoch,
            "planned_steps": planned_steps,
            "steps_done": steps_done,
            "epochs_done": steps_done / steps_per_epoch if steps_per_epoch else 0.0,
            "warmup_steps": warmup,
            "hit_time_cap": callback.hit_cap,
            "train_runtime_s": train_s,
            "sec_per_step_mean": sec_per_step,
            "final_train_loss_mean": float(train_out.training_loss),
            "val_loss_before": float(val_before),
            "val_loss_after": float(val_after),
            "val_loss_examples": len(data.val_sets["eval"]),
            "val_loss_by_set": {
                k: {"before": val_before_all[k], "after": val_after_all[k], "examples": len(data.val_sets[k])}
                for k in data.val_sets
            },
            "cuda_max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else None,
            "cuda_max_reserved_gib": torch.cuda.max_memory_reserved() / 2**30 if dev.type == "cuda" else None,
        },
        "adapter": {
            "dir": str(adapter_dir),
            "weights_sha256": file_sha256(weights) if weights.exists() else None,
            "manifest": adapter_manifest,
        },
        "tracking": {
            "mlflow_tracking_uri": tracker.tracking_uri,
            "mlflow_run_id": tracker.mlflow_run_id,
            "mlflow_error": tracker.error,
        },
        "env": _env(),
        "wall_time_s": time.monotonic() - started,
        "created_at_utc": created.isoformat(timespec="seconds"),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _plot_loss(run_dir / "metrics.jsonl", run_dir / "loss_curve.png")
    for p in ("run_manifest.json", "loss_curve.png", "sft_build_train.json", "metrics.jsonl", "sft_record_ids.jsonl"):
        tracker.artifact(run_dir / p)
    tracker.end(status)
    return manifest
