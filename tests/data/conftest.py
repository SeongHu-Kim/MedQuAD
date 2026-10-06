"""Fixtures for data-pipeline tests (SYNTHETIC rows from data_synthetic.py)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from data_synthetic import SYNTHETIC_ROWS, write_csv

from medquad_qa.data.config import BuildConfig, load_config


@pytest.fixture
def synthetic_root(tmp_path: Path) -> Path:
    """A repo-like root containing synthetic.csv and a config pointing at it (no expected sha)."""
    write_csv(tmp_path / "synthetic.csv", SYNTHETIC_ROWS)
    raw = load_config().model_dump()
    raw["source"]["path"] = "synthetic.csv"
    raw["source"]["expected_sha256"] = None
    raw["source"]["dataset_name"] = "SYNTHETIC test fixture"
    (tmp_path / "build.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    return tmp_path


@pytest.fixture
def synthetic_cfg(synthetic_root: Path) -> BuildConfig:
    return load_config(synthetic_root / "build.yaml")
