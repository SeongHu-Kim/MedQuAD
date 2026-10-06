"""load_generator / adapter sharing / readiness, offline on a tiny Qwen3 + random LoRA adapter."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

from medquad_qa.contracts.interfaces import ArtifactUnavailableError, ChatMessage, ModeUnavailableError
from medquad_qa.contracts.qa import GenerationParams
from medquad_qa.models import ModelSettings, generator_status, load_generator
from medquad_qa.models.factory import ADAPTER_MANIFEST
from medquad_qa.models.testing import build_tiny_qwen

GREEDY = GenerationParams(max_new_tokens=6)
MSGS = [ChatMessage(role="user", content="What are the symptoms of anemia?")]


def _write_adapter(settings: ModelSettings, run_id: str = "run-test", base: tuple[str, str] | None = None) -> Path:
    from peft import LoraConfig, get_peft_model

    model, _ = build_tiny_qwen()
    peft_model = get_peft_model(
        model, LoraConfig(r=4, lora_alpha=8, target_modules="all-linear", init_lora_weights=False)
    )
    adapter_dir = settings.model_dir / "adapters" / run_id
    peft_model.save_pretrained(adapter_dir)
    model_id, revision = base or (settings.generator.model_id, settings.generator.revision)
    (adapter_dir / ADAPTER_MANIFEST).write_text(
        json.dumps({"run_id": run_id, "base_model_id": model_id, "base_revision": revision})
    )
    (settings.model_dir / "adapters" / "CURRENT").write_text(run_id + "\n")
    return adapter_dir


def _last_logits(gen: object, adapter: str | None) -> torch.Tensor:
    backend = gen.backend  # type: ignore[attr-defined]
    ids = torch.tensor([gen.encode(MSGS)])  # type: ignore[attr-defined]
    with backend.lock, backend.activated(adapter) as model, torch.inference_mode():
        return model(ids).logits[0, -1].clone()


def test_finetuned_without_adapter_is_mode_unavailable(tiny_settings: ModelSettings) -> None:
    with pytest.raises(ModeUnavailableError) as err:
        load_generator("finetuned", tiny_settings)
    assert err.value.mode == "finetuned"
    status = generator_status("finetuned", tiny_settings)
    assert not status.ok and not status.required


def test_missing_base_is_artifact_unavailable(tmp_path: Path) -> None:
    settings = ModelSettings.model_validate(
        {"generator": {"model_id": str(tmp_path / "nope"), "revision": "x"}, "device": "cpu"}
    )
    with pytest.raises(ArtifactUnavailableError):
        load_generator("base", settings)
    status = generator_status("base", settings)
    assert not status.ok and status.required


def test_adapter_shares_backend_and_keeps_base_intact(tiny_settings: ModelSettings) -> None:
    base = load_generator("base", tiny_settings)
    before = base.generate(MSGS, GREEDY)
    base_logits = _last_logits(base, None)

    _write_adapter(tiny_settings)
    tuned = load_generator("finetuned", tiny_settings)
    assert tuned.backend is base.backend
    assert tuned.model_version == f"{tiny_settings.generator.base_version}+lora:run-test"

    after = base.generate(MSGS, GREEDY)
    assert (after.text, after.completion_tokens) == (before.text, before.completion_tokens)
    assert torch.equal(_last_logits(base, None), base_logits)
    assert not torch.allclose(_last_logits(tuned, "run-test"), base_logits)
    assert generator_status("finetuned", tiny_settings).ok


def test_interleaved_threads_match_sequential(tiny_settings: ModelSettings) -> None:
    _write_adapter(tiny_settings)
    base, tuned = load_generator("base", tiny_settings), load_generator("finetuned", tiny_settings)
    expected = [base.generate(MSGS, GREEDY).text, tuned.generate(MSGS, GREEDY).text]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(g.generate, MSGS, GREEDY) for _ in range(4) for g in (base, tuned)]
        texts = [f.result().text for f in futures]
    assert texts == expected * 4


def test_adapter_for_other_base_is_rejected(tiny_settings: ModelSettings) -> None:
    _write_adapter(tiny_settings, base=("Qwen/other", "abc"))
    with pytest.raises(ModeUnavailableError, match="adapter base"):
        load_generator("finetuned", tiny_settings)
    assert not generator_status("finetuned", tiny_settings).ok


def test_base_status_ok_for_local_model(tiny_settings: ModelSettings) -> None:
    status = generator_status("base", tiny_settings)
    assert status.ok and status.name == "generator:base" and status.version == tiny_settings.generator.base_version
