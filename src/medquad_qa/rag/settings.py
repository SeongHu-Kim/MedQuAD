"""RAG pipeline settings (plain pydantic; no api extra needed; D-024)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from medquad_qa.retrieval.settings import RetrievalSettings

GateMode = Literal["auto", "predictor", "heuristic", "off"]


class RagSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    gate_mode: GateMode = "auto"
    gate_heuristic_path: Path = Path("configs/retrieval/gate_heuristic.json")
    max_input_tokens: int = Field(default=3072, ge=256, description="Used when the generator exposes none.")
    preload_generators: tuple[Literal["base", "finetuned"], ...] = ("base", "finetuned")

    @classmethod
    def from_env(cls, **overrides: object) -> RagSettings:
        values: dict[str, object] = {"retrieval": RetrievalSettings.from_env()}
        env = os.environ
        if env.get("MEDQUAD_GATE_MODE"):
            values["gate_mode"] = env["MEDQUAD_GATE_MODE"].strip()
        if env.get("MEDQUAD_GATE_HEURISTIC_PATH"):
            values["gate_heuristic_path"] = Path(env["MEDQUAD_GATE_HEURISTIC_PATH"].strip())
        values.update(overrides)
        return cls.model_validate(values)
