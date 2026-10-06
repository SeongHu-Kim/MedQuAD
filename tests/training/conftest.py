"""Fixtures for training tests (synthetic data only)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from training_fixtures import make_split, write_jsonl

from medquad_qa.models.factory import clear_backends
from medquad_qa.models.testing import build_tiny_tokenizer, save_tiny_qwen


@pytest.fixture
def exports(tmp_path: Path) -> dict[str, Path]:
    return {
        "train": write_jsonl(make_split("train", 0), tmp_path / "records_train.jsonl"),
        "val": write_jsonl(make_split("val", 100), tmp_path / "records_val.jsonl"),
        "test": write_jsonl(make_split("test", 200), tmp_path / "records_test.jsonl"),
    }


@pytest.fixture(scope="session")
def tiny_tokenizer() -> object:
    return build_tiny_tokenizer()


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return save_tiny_qwen(tmp_path_factory.mktemp("tiny-qwen-train"))


@pytest.fixture(autouse=True)
def _fresh_backends() -> Iterator[None]:
    clear_backends()
    yield
    clear_backends()
