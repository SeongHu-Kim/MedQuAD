from __future__ import annotations

from pathlib import Path

import pytest

from medquad_qa.contracts.interfaces import (
    BatchGenerator,
    ChatMessage,
    GenerationError,
    GenerationTimeoutError,
    Generator,
)
from medquad_qa.contracts.qa import GenerationParams
from medquad_qa.models import FakeGenerator
from medquad_qa.models.settings import DEFAULT_GENERATOR_CONFIG_PATH, GeneratorConfig, ModelSettings

REPO = Path(__file__).resolve().parents[2]


def test_yaml_matches_code_defaults() -> None:
    assert GeneratorConfig.from_yaml(REPO / DEFAULT_GENERATOR_CONFIG_PATH) == GeneratorConfig()


def test_approved_generation_settings() -> None:
    cfg = GeneratorConfig()
    assert cfg.base_version == "Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554"
    assert (cfg.max_input_tokens, cfg.timeout_s, cfg.repetition_penalty) == (3072, 60.0, 1.05)
    params = GenerationParams()
    assert (params.temperature, params.max_new_tokens, params.top_p) == (0.0, 256, 1.0)


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)  # no configs/ here -> code defaults
    monkeypatch.setenv("MEDQUAD_DEVICE", "cpu")
    monkeypatch.setenv("MEDQUAD_MODEL_DIR", "/x/models")
    monkeypatch.setenv("MEDQUAD_ADAPTER_DIR", "/x/adapter")
    monkeypatch.delenv("MEDQUAD_ALLOW_MODEL_DOWNLOAD", raising=False)
    s = ModelSettings.from_env()
    assert s.device == "cpu" and s.model_dir == Path("/x/models") and s.adapter_dir == Path("/x/adapter")
    assert s.local_files_only is True
    assert s.generator == GeneratorConfig()


def _msgs(text: str) -> list[ChatMessage]:
    return [ChatMessage(role="system", content="sys"), ChatMessage(role="user", content=text)]


def test_fake_conforms_and_cites() -> None:
    fake = FakeGenerator()
    assert isinstance(fake, Generator) and isinstance(fake, BatchGenerator)
    out = fake.generate(_msgs('<evidence id="E2">text</evidence> question?'), GenerationParams())
    assert "[E2]" in out.text and out.finish_reason == "stop"
    assert fake.generate(_msgs("no evidence"), GenerationParams()).text == "Synthetic fixture answer."
    assert len(fake.generate_batch([_msgs("a"), _msgs("b")], GenerationParams())) == 2


def test_fake_failures_and_bounds() -> None:
    with pytest.raises(GenerationTimeoutError):
        FakeGenerator(fail_with="timeout").generate(_msgs("q"), GenerationParams())
    with pytest.raises(GenerationError):
        FakeGenerator(fail_with="error").generate(_msgs("q"), GenerationParams())
    with pytest.raises(GenerationError, match="max_input_tokens"):
        FakeGenerator(max_input_tokens=2).generate(_msgs("one two three"), GenerationParams())
    out = FakeGenerator(lambda m: "w " * 10).generate(_msgs("q"), GenerationParams(max_new_tokens=3))
    assert out.finish_reason == "length" and out.completion_tokens == 3


def test_load_answerability_predictor_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from medquad_qa.contracts.interfaces import ArtifactUnavailableError
    from medquad_qa.models import load_answerability_predictor

    monkeypatch.delenv("MEDQUAD_ANSWERABILITY_DIR", raising=False)
    with pytest.raises(ArtifactUnavailableError):
        load_answerability_predictor(ModelSettings(model_dir=tmp_path))
