"""Real-weights tests (excluded from `make test`). Require the pinned Qwen3-4B snapshot in the HF cache:
  .venv/bin/python scripts/training/download_model.py
Run: .venv/bin/pytest tests/models/test_models_real_generator.py -m "real_model" -q
"""

from __future__ import annotations

import pytest
import torch

from medquad_qa.contracts.interfaces import ChatMessage
from medquad_qa.contracts.qa import GenerationParams
from medquad_qa.models import ModelSettings, generator_status, load_generator

pytestmark = pytest.mark.real_model

MSGS = [
    ChatMessage(role="system", content="You provide general medical information. Answer in two sentences."),
    ChatMessage(role="user", content="What is high blood pressure?"),
]


@pytest.fixture
def settings() -> ModelSettings:
    s = ModelSettings()
    if not generator_status("base", s).ok:
        pytest.skip("pinned base weights not in the local HF cache")
    return s


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_real_generation_cuda(settings: ModelSettings) -> None:
    gen = load_generator("base", settings.model_copy(update={"device": "cuda"}))
    assert next(gen.backend.model.parameters()).dtype == torch.bfloat16
    out = gen.generate(MSGS, GenerationParams(max_new_tokens=96))
    assert out.text and out.completion_tokens > 0 and out.prompt_tokens == gen.count_tokens(MSGS)
    assert "<|im_" not in out.text and "<think>" not in out.text
    assert out.model_version == settings.generator.base_version
    again = gen.generate(MSGS, GenerationParams(max_new_tokens=96))
    assert again.text == out.text  # greedy, same hardware and batch shape


@pytest.mark.slow
def test_real_generation_cpu_smoke(settings: ModelSettings) -> None:
    gen = load_generator("base", settings.model_copy(update={"device": "cpu"}))
    assert next(gen.backend.model.parameters()).dtype == torch.float32
    out = gen.generate(MSGS, GenerationParams(max_new_tokens=12))
    assert out.text and 0 < out.completion_tokens <= 12


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_real_finetuned_adapter_cuda(settings: ModelSettings) -> None:
    status = generator_status("finetuned", settings)
    if not status.ok:
        pytest.skip(f"no promoted adapter: {status.detail}")
    cuda = settings.model_copy(update={"device": "cuda"})
    base, tuned = load_generator("base", cuda), load_generator("finetuned", cuda)
    assert tuned.backend is base.backend and tuned.model_version == status.version
    b = base.generate(MSGS, GenerationParams(max_new_tokens=48))
    t = tuned.generate(MSGS, GenerationParams(max_new_tokens=48))
    assert t.text and t.text != b.text
    assert base.generate(MSGS, GenerationParams(max_new_tokens=48)).text == b.text  # base unaffected by the adapter
