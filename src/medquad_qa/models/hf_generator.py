"""Hugging Face causal-LM generator (implements ``Generator`` and ``BatchGenerator``).

One ``GeneratorBackend`` owns the model weights and tokenizer. Base and fine-tuned ``HFGenerator`` views share
it: after a LoRA adapter is attached the backend holds a ``PeftModel`` and each call either activates the
adapter or runs inside ``disable_adapter()``. A single lock serialises all generation on the backend, so the
adapter toggle can never interleave with another request.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList

from medquad_qa.contracts.interfaces import ChatMessage, GenerationError, GenerationResult, GenerationTimeoutError
from medquad_qa.contracts.qa import GenerationParams
from medquad_qa.models.settings import GeneratorConfig


class _DeadlineCriteria(StoppingCriteria):
    """Stops generation once the wall-clock deadline passes and remembers that it did."""

    def __init__(self, deadline: float) -> None:
        self.deadline = deadline
        self.fired = False

    def __call__(self, input_ids: torch.LongTensor, scores: Any, **kwargs: Any) -> torch.BoolTensor:
        if time.monotonic() >= self.deadline:
            self.fired = True
        return torch.full((input_ids.shape[0],), self.fired, dtype=torch.bool, device=input_ids.device)  # type: ignore[return-value]


class GeneratorBackend:
    """Shared model + tokenizer + lock. Adapters are attached by name (the training run id)."""

    def __init__(self, model: Any, tokenizer: Any, device: torch.device) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.lock = threading.Lock()
        self.adapters: dict[str, str] = {}  # adapter name -> source path
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        eos = self.model.generation_config.eos_token_id
        self.eos_token_ids: set[int] = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
        if self.tokenizer.eos_token_id is not None:
            self.eos_token_ids.add(int(self.tokenizer.eos_token_id))

    def attach_adapter(self, name: str, path: str) -> None:
        """Load a LoRA adapter (inference only). Idempotent per name."""
        from peft import PeftModel

        with self.lock:
            if name in self.adapters:
                return
            if isinstance(self.model, PeftModel):
                self.model.load_adapter(path, adapter_name=name, is_trainable=False)
            else:
                self.model = PeftModel.from_pretrained(self.model, path, adapter_name=name, is_trainable=False)
            self.model.eval()
            self.adapters[name] = path

    @contextmanager
    def activated(self, adapter_name: str | None) -> Iterator[Any]:
        """Yield the model with ``adapter_name`` active (None = base weights). Caller must hold ``lock``."""
        from peft import PeftModel

        if not isinstance(self.model, PeftModel):
            if adapter_name is not None:
                raise GenerationError(f"adapter '{adapter_name}' is not attached")
            yield self.model
        elif adapter_name is None:
            with self.model.disable_adapter():
                yield self.model
        else:
            self.model.set_adapter(adapter_name)
            yield self.model


class HFGenerator:
    """Chat generator over a shared backend. ``adapter_name=None`` is the base model."""

    def __init__(
        self,
        backend: GeneratorBackend,
        config: GeneratorConfig,
        *,
        model_version: str,
        adapter_name: str | None = None,
    ) -> None:
        self.backend = backend
        self.config = config
        self.model_version = model_version
        self.adapter_name = adapter_name

    @property
    def max_input_tokens(self) -> int:
        return self.config.max_input_tokens

    # ------------------------------------------------------------------ tokens
    def encode(self, messages: list[ChatMessage]) -> list[int]:
        """Prompt token ids after the chat template (with the assistant generation prompt)."""
        ids = self.backend.tokenizer.apply_chat_template(
            [m.model_dump() for m in messages], add_generation_prompt=True, tokenize=True, return_dict=False
        )
        if isinstance(ids, dict) or hasattr(ids, "keys"):  # some tokenizers return a BatchEncoding
            ids = ids["input_ids"]
        return [int(t) for t in ids]

    def count_tokens(self, messages: list[ChatMessage]) -> int:
        """Exact prompt length in tokens, as checked against ``max_input_tokens`` (D-022)."""
        return len(self.encode(messages))

    # ------------------------------------------------------------------ generation
    def generate(self, messages: list[ChatMessage], params: GenerationParams) -> GenerationResult:
        return self._run([messages], params, timeout_s=self.config.timeout_s)[0]

    def generate_batch(self, batch: list[list[ChatMessage]], params: GenerationParams) -> list[GenerationResult]:
        """Left-padded batched generation. The deadline scales with batch size (timeout_s per item).
        Greedy outputs can differ slightly from unbatched calls because of padding and kernel choice."""
        if not batch:
            return []
        return self._run(batch, params, timeout_s=self.config.timeout_s * len(batch))

    def _generation_config(self, params: GenerationParams) -> GenerationConfig:
        tok = self.backend.tokenizer
        kwargs: dict[str, Any] = {
            "max_new_tokens": params.max_new_tokens,
            "repetition_penalty": self.config.repetition_penalty,
            "eos_token_id": sorted(self.backend.eos_token_ids) or None,
            "pad_token_id": tok.pad_token_id,
        }
        if params.temperature == 0.0:
            kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        else:
            kwargs.update(do_sample=True, temperature=params.temperature, top_p=params.top_p, top_k=0)
        return GenerationConfig(**kwargs)

    def _run(
        self, batch: list[list[ChatMessage]], params: GenerationParams, timeout_s: float
    ) -> list[GenerationResult]:
        start = time.monotonic()
        deadline = start + timeout_s
        prompts = [self.encode(m) for m in batch]
        for i, ids in enumerate(prompts):
            if len(ids) > self.config.max_input_tokens:
                raise GenerationError(
                    f"prompt has {len(ids)} tokens > max_input_tokens={self.config.max_input_tokens} (item {i})"
                )

        if not self.backend.lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
            raise GenerationTimeoutError(f"generator busy for more than {timeout_s:.1f}s")
        try:
            criteria = _DeadlineCriteria(deadline)
            input_ids, attention_mask = self._left_pad(prompts)
            gen_config = self._generation_config(params)
            if params.seed is not None:
                torch.manual_seed(params.seed)
            try:
                with self.backend.activated(self.adapter_name) as model, torch.inference_mode():
                    out = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        generation_config=gen_config,
                        stopping_criteria=StoppingCriteriaList([criteria]),
                    )
            except GenerationError:
                raise
            except Exception as exc:  # CUDA/kernel/OOM errors surface as 502, with the original as cause
                raise GenerationError(f"generation failed: {type(exc).__name__}: {exc}") from exc
        finally:
            self.backend.lock.release()

        if criteria.fired:
            raise GenerationTimeoutError(f"generation exceeded {timeout_s:.1f}s")
        latency_ms = (time.monotonic() - start) * 1000.0
        new_tokens = out[:, input_ids.shape[1] :].tolist()
        return [self._result(row, len(p), latency_ms, params) for row, p in zip(new_tokens, prompts, strict=True)]

    def _left_pad(self, prompts: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        pad_id = int(self.backend.tokenizer.pad_token_id)
        width = max(len(p) for p in prompts)
        ids = [[pad_id] * (width - len(p)) + p for p in prompts]
        mask = [[0] * (width - len(p)) + [1] * len(p) for p in prompts]
        device = self.backend.device
        return torch.tensor(ids, dtype=torch.long, device=device), torch.tensor(mask, dtype=torch.long, device=device)

    def _result(
        self, tokens: list[int], prompt_len: int, latency_ms: float, params: GenerationParams
    ) -> GenerationResult:
        eos_at = next((i for i, t in enumerate(tokens) if t in self.backend.eos_token_ids), None)
        if eos_at is None:
            content, completion, finish = tokens, len(tokens), "length"
        else:
            content, completion, finish = tokens[:eos_at], eos_at + 1, "stop"
        text = self.backend.tokenizer.decode(content, skip_special_tokens=True).strip()
        return GenerationResult(
            text=text,
            model_version=self.model_version,
            prompt_tokens=prompt_len,
            completion_tokens=completion,
            latency_ms=latency_ms,
            finish_reason=finish,
        )
