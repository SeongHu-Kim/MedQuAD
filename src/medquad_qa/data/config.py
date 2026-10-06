"""Typed loader for configs/data/build.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "data" / "build.yaml"
DEFAULT_QTYPE_CONFIG = REPO_ROOT / "configs" / "data" / "question_types.yaml"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SourceConfig(_Frozen):
    path: str
    dataset_name: str
    expected_columns: list[str]
    expected_sha256: str | None = None
    kaggle_dataset_ref: str | None = None


class OutputConfig(_Frozen):
    processed_dir: str
    manifests_dir: str


class ExclusionConfig(_Frozen):
    heading_max_words: int = 8


class QualityConfig(_Frozen):
    short_answer_max_words: int = 10
    repeated_bullet_min_chars: int = 15


class GroupingConfig(_Frozen):
    boilerplate_min_topics: int = 3
    shingle_size: int = 5
    rare_shingle_max_df: int = 10
    min_shared_rare_shingles: int = 5
    near_dup_jaccard: float = 0.8
    residual_jaccard_min: float = 0.5


class SplitConfig(_Frozen):
    seed: int
    ratios: dict[str, float] = Field(description="Keys: train, validation, test.")


class BuildConfig(_Frozen):
    pipeline_version: str
    corpus_semver: str
    source: SourceConfig
    outputs: OutputConfig
    exclusion: ExclusionConfig = ExclusionConfig()
    quality: QualityConfig = QualityConfig()
    grouping: GroupingConfig = GroupingConfig()
    split: SplitConfig


def load_config(path: Path = DEFAULT_CONFIG) -> BuildConfig:
    with path.open(encoding="utf-8") as fh:
        return BuildConfig.model_validate(yaml.safe_load(fh))
