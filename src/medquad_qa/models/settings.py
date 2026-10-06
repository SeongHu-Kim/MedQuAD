"""Generator configuration and runtime settings.

``GeneratorConfig`` holds the approved, versioned generation settings (D-011, D-012). Its defaults mirror
``configs/models/generator.yaml``; a test keeps the two in sync. ``ModelSettings`` adds deployment knobs
(device, artifact directories, download policy) read from ``MEDQUAD_*`` environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_GENERATOR_CONFIG_PATH = Path("configs/models/generator.yaml")

Device = Literal["auto", "cuda", "cpu"]
DTypeName = Literal["bfloat16", "float16", "float32"]


class GeneratorConfig(BaseModel):
    """Approved generator settings. Matched across all four experiment modes (D-012)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    revision: str = "cdbee75f17c01a7cc42f958dc650907174af0554"
    max_input_tokens: int = Field(default=3072, ge=1, description="Prompt longer than this raises GenerationError.")
    timeout_s: float = Field(default=60.0, gt=0.0, description="Internal deadline per generate() call.")
    repetition_penalty: float = Field(default=1.05, ge=1.0)
    attn_implementation: str = "sdpa"
    cuda_dtype: DTypeName = "bfloat16"
    cpu_dtype: DTypeName = "float32"

    @property
    def base_version(self) -> str:
        return f"{self.model_id}@{self.revision}"

    @classmethod
    def from_yaml(cls, path: str | Path) -> GeneratorConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(**data)


class ModelSettings(BaseModel):
    """Runtime settings for loading generators and adapters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    device: Device = "auto"
    model_dir: Path = Path("artifacts/models")
    adapter_dir: Path | None = Field(
        default=None, description="LoRA adapter directory. None -> <model_dir>/adapters/<run id in CURRENT>."
    )
    local_files_only: bool = Field(
        default=True, description="Never download at load time; use scripts/training/download_model.py first."
    )

    @classmethod
    def from_env(cls) -> ModelSettings:
        """Build settings from MEDQUAD_* environment variables (see docs/models/inference.md)."""
        cfg_path = os.environ.get("MEDQUAD_GENERATOR_CONFIG")
        if cfg_path:
            generator = GeneratorConfig.from_yaml(cfg_path)
        elif DEFAULT_GENERATOR_CONFIG_PATH.is_file():
            generator = GeneratorConfig.from_yaml(DEFAULT_GENERATOR_CONFIG_PATH)
        else:
            generator = GeneratorConfig()
        adapter_dir = os.environ.get("MEDQUAD_ADAPTER_DIR")
        return cls(
            generator=generator,
            device=os.environ.get("MEDQUAD_DEVICE", "auto"),
            model_dir=Path(os.environ.get("MEDQUAD_MODEL_DIR", "artifacts/models")),
            adapter_dir=Path(adapter_dir) if adapter_dir else None,
            local_files_only=os.environ.get("MEDQUAD_ALLOW_MODEL_DOWNLOAD", "false").lower() not in ("1", "true"),
        )
