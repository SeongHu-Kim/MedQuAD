from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from medquad_qa.models.factory import clear_backends
from medquad_qa.models.settings import GeneratorConfig, ModelSettings
from medquad_qa.models.testing import save_tiny_qwen


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return save_tiny_qwen(tmp_path_factory.mktemp("tiny-qwen"))


@pytest.fixture(autouse=True)
def _fresh_backends() -> Iterator[None]:
    clear_backends()
    yield
    clear_backends()


@pytest.fixture
def tiny_settings(tiny_model_dir: Path, tmp_path: Path) -> ModelSettings:
    return ModelSettings(
        generator=GeneratorConfig(model_id=str(tiny_model_dir), revision="local"),
        device="cpu",
        model_dir=tmp_path / "models",
    )
