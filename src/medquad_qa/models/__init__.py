"""medquad_qa.models: generator adapters (HF base / +LoRA), fakes, and answerability predictors.

Heavy imports (torch, transformers, peft) happen only when a real generator is requested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from medquad_qa.models.fake import FakeGenerator
from medquad_qa.models.settings import GeneratorConfig, ModelSettings

if TYPE_CHECKING:
    from medquad_qa.contracts.interfaces import ComponentStatus
    from medquad_qa.models.answerability import LexicalAnswerabilityPredictor
    from medquad_qa.models.hf_generator import HFGenerator

__all__ = [
    "FakeGenerator",
    "GeneratorConfig",
    "ModelSettings",
    "generator_status",
    "load_answerability_predictor",
    "load_generator",
]


def load_generator(variant: Any, settings: ModelSettings | None = None) -> HFGenerator:
    """See :func:`medquad_qa.models.factory.load_generator`."""
    from medquad_qa.models.factory import load_generator as _load

    return _load(variant, settings)


def generator_status(variant: Any, settings: ModelSettings | None = None) -> ComponentStatus:
    """See :func:`medquad_qa.models.factory.generator_status`."""
    from medquad_qa.models.factory import generator_status as _status

    return _status(variant, settings)


def load_answerability_predictor(settings: ModelSettings | None = None) -> LexicalAnswerabilityPredictor:
    """Serving answerability predictor (lexical LR, numpy only; no sklearn/TF at runtime).

    Directory: ``MEDQUAD_ANSWERABILITY_DIR`` or ``<model_dir>/answerability/lexical_lr``.
    Raises ArtifactUnavailableError if model.json/threshold.json are missing.
    """
    import os
    from pathlib import Path

    from medquad_qa.models.answerability import LexicalAnswerabilityPredictor

    settings = settings or ModelSettings.from_env()
    env = os.environ.get("MEDQUAD_ANSWERABILITY_DIR")
    directory = Path(env) if env else settings.model_dir / "answerability" / "lexical_lr"
    return LexicalAnswerabilityPredictor.load(directory)
