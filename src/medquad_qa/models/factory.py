"""``load_generator(variant, settings)``: the single entry point used by the RAG pipeline, API and evaluation.

Both variants share one cached ``GeneratorBackend`` per (model, revision, device, dtype), so serving base and
fine-tuned modes costs one copy of the weights. Fine-tuned adapters are resolved from
``<model_dir>/adapters/CURRENT`` (a run id) unless ``settings.adapter_dir`` is set, and must carry an
``adapter_manifest.json`` whose base model/revision match the configured generator.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Literal

import torch

from medquad_qa.contracts.interfaces import ArtifactUnavailableError, ComponentStatus, ModeUnavailableError
from medquad_qa.models.hf_generator import GeneratorBackend, HFGenerator
from medquad_qa.models.settings import GeneratorConfig, ModelSettings
from medquad_qa.models.torch_runtime import apply_native_jit_guard

Variant = Literal["base", "finetuned"]
ADAPTER_MANIFEST = "adapter_manifest.json"

_BACKENDS: dict[tuple[str, str, str, str], GeneratorBackend] = {}
_BACKENDS_LOCK = threading.Lock()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise ArtifactUnavailableError("MEDQUAD_DEVICE=cuda but CUDA is not available")
    return torch.device(device)


def _dtype(config: GeneratorConfig, device: torch.device) -> torch.dtype:
    name = config.cuda_dtype if device.type == "cuda" else config.cpu_dtype
    return getattr(torch, name)


def get_backend(settings: ModelSettings) -> GeneratorBackend:
    """Load (once) and return the shared backend for these settings."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = settings.generator
    apply_native_jit_guard()
    device = resolve_device(settings.device)
    dtype = _dtype(cfg, device)
    key = (cfg.model_id, cfg.revision, str(device), str(dtype))
    with _BACKENDS_LOCK:
        if key in _BACKENDS:
            return _BACKENDS[key]
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                cfg.model_id, revision=cfg.revision, local_files_only=settings.local_files_only
            )
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_id,
                revision=cfg.revision,
                dtype=dtype,
                attn_implementation=cfg.attn_implementation,
                local_files_only=settings.local_files_only,
                # mmap-backed weights copy to the GB10 GPU at ~150 MB/s (page faults); reading the files into
                # memory first loads in ~5 s instead of ~50 s (docs/models/inference.md)
                disable_mmap=True,
            )
        except (OSError, ValueError) as exc:
            raise ArtifactUnavailableError(f"cannot load generator {cfg.base_version}: {exc}") from exc
        model.to(device).eval()  # type: ignore[arg-type]
        backend = GeneratorBackend(model, tokenizer, device)
        _BACKENDS[key] = backend
        return backend


def clear_backends() -> None:
    """Drop cached backends (tests, or to free memory)."""
    with _BACKENDS_LOCK:
        _BACKENDS.clear()


def resolve_adapter_dir(settings: ModelSettings) -> Path | None:
    if settings.adapter_dir is not None:
        return settings.adapter_dir
    pointer = settings.model_dir / "adapters" / "CURRENT"
    if not pointer.is_file():
        return None
    run_id = pointer.read_text(encoding="utf-8").strip()
    return settings.model_dir / "adapters" / run_id if run_id else None


def read_adapter_manifest(adapter_dir: Path, config: GeneratorConfig) -> dict[str, Any]:
    """Validate the adapter directory against the configured base model. Raises ModeUnavailableError."""
    manifest_path = adapter_dir / ADAPTER_MANIFEST
    if not (adapter_dir / "adapter_config.json").is_file() or not manifest_path.is_file():
        raise ModeUnavailableError("finetuned", f"no adapter at {adapter_dir}")
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = (manifest.get("base_model_id"), manifest.get("base_revision"))
    if base != (config.model_id, config.revision):
        raise ModeUnavailableError("finetuned", f"adapter base {base} != configured {config.base_version}")
    if not manifest.get("run_id"):
        raise ModeUnavailableError("finetuned", "adapter manifest has no run_id")
    return manifest


def load_generator(variant: Variant, settings: ModelSettings | None = None) -> HFGenerator:
    """Return a generator for ``variant``. Raises ArtifactUnavailableError (base weights missing) or
    ModeUnavailableError (no valid adapter for 'finetuned')."""
    settings = settings or ModelSettings.from_env()
    cfg = settings.generator
    if variant == "base":
        return HFGenerator(get_backend(settings), cfg, model_version=cfg.base_version)
    if variant != "finetuned":
        raise ValueError(f"unknown generator variant {variant!r}")
    adapter_dir = resolve_adapter_dir(settings)
    if adapter_dir is None:
        raise ModeUnavailableError("finetuned", "no adapter configured (adapters/CURRENT missing)")
    run_id = str(read_adapter_manifest(adapter_dir, cfg)["run_id"])
    backend = get_backend(settings)
    backend.attach_adapter(run_id, str(adapter_dir))
    return HFGenerator(backend, cfg, model_version=f"{cfg.base_version}+lora:{run_id}", adapter_name=run_id)


def generator_status(variant: Variant, settings: ModelSettings | None = None) -> ComponentStatus:
    """Cheap readiness check from files only (no weights loaded). Never raises."""
    settings = settings or ModelSettings.from_env()
    cfg = settings.generator
    name = f"generator:{variant}"
    try:
        if not _base_files_present(cfg):
            return ComponentStatus(name=name, ok=False, required=variant == "base", detail="base weights not found")
        if variant == "base":
            return ComponentStatus(name=name, ok=True, required=True, version=cfg.base_version)
        adapter_dir = resolve_adapter_dir(settings)
        if adapter_dir is None:
            return ComponentStatus(name=name, ok=False, required=False, detail="no adapter configured")
        run_id = read_adapter_manifest(adapter_dir, cfg)["run_id"]
        return ComponentStatus(name=name, ok=True, required=False, version=f"{cfg.base_version}+lora:{run_id}")
    except Exception as exc:  # readiness must never raise
        return ComponentStatus(name=name, ok=False, required=variant == "base", detail=str(exc)[:200])


def _base_files_present(cfg: GeneratorConfig) -> bool:
    local = Path(cfg.model_id)
    if local.is_dir():
        return (local / "config.json").is_file()
    from huggingface_hub import try_to_load_from_cache

    hit = try_to_load_from_cache(cfg.model_id, "model.safetensors.index.json", revision=cfg.revision)
    return isinstance(hit, str)
