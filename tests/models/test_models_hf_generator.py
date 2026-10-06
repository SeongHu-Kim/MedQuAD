"""Offline tests of HFGenerator on a tiny random-init Qwen3 (CPU, fp32)."""

from __future__ import annotations

import pytest

from medquad_qa.contracts.interfaces import (
    BatchGenerator,
    ChatMessage,
    GenerationError,
    GenerationTimeoutError,
    Generator,
)
from medquad_qa.contracts.qa import GenerationParams
from medquad_qa.models import ModelSettings, load_generator
from medquad_qa.models.hf_generator import HFGenerator

GREEDY = GenerationParams(max_new_tokens=8)


def _msgs(text: str = "What are the symptoms of anemia?") -> list[ChatMessage]:
    return [ChatMessage(role="system", content="Use only the evidence."), ChatMessage(role="user", content=text)]


def _with_config(gen: HFGenerator, **overrides: object) -> HFGenerator:
    cfg = gen.config.model_copy(update=overrides)
    return HFGenerator(gen.backend, cfg, model_version=gen.model_version, adapter_name=gen.adapter_name)


def test_protocols_and_version(tiny_settings: ModelSettings) -> None:
    gen = load_generator("base", tiny_settings)
    assert isinstance(gen, Generator) and isinstance(gen, BatchGenerator)
    assert gen.model_version == f"{tiny_settings.generator.model_id}@local"


def test_count_tokens_matches_prompt_tokens(tiny_settings: ModelSettings) -> None:
    gen = load_generator("base", tiny_settings)
    out = gen.generate(_msgs(), GREEDY)
    assert out.prompt_tokens == gen.count_tokens(_msgs()) > 0
    assert out.model_version == gen.model_version


def test_greedy_is_repeatable(tiny_settings: ModelSettings) -> None:
    gen = load_generator("base", tiny_settings)
    a, b = gen.generate(_msgs(), GREEDY), gen.generate(_msgs(), GREEDY)
    assert (a.text, a.completion_tokens, a.finish_reason) == (b.text, b.completion_tokens, b.finish_reason)
    assert a.finish_reason == "length" and a.completion_tokens == 8


def test_stop_finish_reason_when_eos_emitted(tiny_settings: ModelSettings) -> None:
    gen = load_generator("base", tiny_settings)
    first = gen.generate(_msgs(), GenerationParams(max_new_tokens=1))
    assert first.finish_reason == "length"
    # make whatever greedy emits first an EOS token: generation must stop after one token with reason 'stop'
    ids = gen.encode(_msgs())
    import torch

    with torch.inference_mode():
        logits = gen.backend.model(torch.tensor([ids])).logits[0, -1]
    gen.backend.eos_token_ids.add(int(logits.argmax()))
    out = gen.generate(_msgs(), GREEDY)
    assert out.finish_reason == "stop" and out.completion_tokens == 1


def test_overflow_raises_never_truncates(tiny_settings: ModelSettings) -> None:
    gen = load_generator("base", tiny_settings)
    small = _with_config(gen, max_input_tokens=gen.count_tokens(_msgs()) - 1)
    with pytest.raises(GenerationError, match="max_input_tokens"):
        small.generate(_msgs(), GREEDY)
    with pytest.raises(GenerationError, match="max_input_tokens"):
        small.generate_batch([_msgs("hi"), _msgs()], GREEDY)


def test_deadline_raises_timeout(tiny_settings: ModelSettings) -> None:
    gen = _with_config(load_generator("base", tiny_settings), timeout_s=1e-9)
    with pytest.raises(GenerationTimeoutError):
        gen.generate(_msgs(), GREEDY)


def test_busy_lock_raises_timeout(tiny_settings: ModelSettings) -> None:
    gen = _with_config(load_generator("base", tiny_settings), timeout_s=0.05)
    gen.backend.lock.acquire()
    try:
        with pytest.raises(GenerationTimeoutError, match="busy"):
            gen.generate(_msgs(), GREEDY)
    finally:
        gen.backend.lock.release()


def test_runtime_failure_maps_to_generation_error(
    tiny_settings: ModelSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    gen = load_generator("base", tiny_settings)

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("CUDA error: synthetic")

    monkeypatch.setattr(gen.backend.model, "generate", boom)
    with pytest.raises(GenerationError, match="synthetic"):
        gen.generate(_msgs(), GREEDY)
    assert not gen.backend.lock.locked()


def test_batch_matches_single_greedy(tiny_settings: ModelSettings) -> None:
    gen = load_generator("base", tiny_settings)
    prompts = [_msgs(), _msgs("How is high blood pressure treated? Use lifestyle changes."), _msgs("hi")]
    singles = [gen.generate(p, GREEDY) for p in prompts]
    batch = gen.generate_batch(prompts, GREEDY)
    assert [(r.text, r.prompt_tokens, r.completion_tokens) for r in batch] == [
        (r.text, r.prompt_tokens, r.completion_tokens) for r in singles
    ]
    assert gen.generate_batch([], GREEDY) == []


def test_sampling_with_seed_is_repeatable(tiny_settings: ModelSettings) -> None:
    gen = load_generator("base", tiny_settings)
    params = GenerationParams(max_new_tokens=8, temperature=1.0, top_p=0.9, seed=7)
    assert gen.generate(_msgs(), params).text == gen.generate(_msgs(), params).text
