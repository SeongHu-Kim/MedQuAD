"""Deterministic test double for ``Generator``/``BatchGenerator``. Synthetic output only; never a model result."""

from __future__ import annotations

import re
from collections.abc import Callable

from medquad_qa.contracts.interfaces import ChatMessage, GenerationError, GenerationResult, GenerationTimeoutError
from medquad_qa.contracts.qa import GenerationParams

_EVIDENCE_LABEL = re.compile(r'<evidence id="(E\d+)"')


def default_responder(messages: list[ChatMessage]) -> str:
    """Cites the first evidence label if the prompt has one; otherwise a plain synthetic sentence."""
    prompt = "\n".join(m.content for m in messages)
    label = _EVIDENCE_LABEL.search(prompt)
    if label:
        return f"Synthetic fixture answer based on the supplied evidence [{label.group(1)}]."
    return "Synthetic fixture answer."


class FakeGenerator:
    """Configurable fake. Token counts are whitespace word counts.

    ``fail_with`` = "timeout" raises GenerationTimeoutError, "error" raises GenerationError.
    ``max_input_tokens`` mimics the real overflow check.
    """

    def __init__(
        self,
        responder: Callable[[list[ChatMessage]], str] | None = None,
        *,
        model_version: str = "fake-generator@0",
        fail_with: str | None = None,
        max_input_tokens: int = 3072,
        latency_ms: float = 1.0,
    ) -> None:
        self.responder = responder or default_responder
        self.model_version = model_version
        self.fail_with = fail_with
        self.max_input_tokens = max_input_tokens
        self.latency_ms = latency_ms
        self.calls: list[list[ChatMessage]] = []

    def count_tokens(self, messages: list[ChatMessage]) -> int:
        return sum(len(m.content.split()) for m in messages)

    def generate(self, messages: list[ChatMessage], params: GenerationParams) -> GenerationResult:
        self.calls.append(messages)
        if self.fail_with == "timeout":
            raise GenerationTimeoutError("fake timeout")
        if self.fail_with == "error":
            raise GenerationError("fake failure")
        prompt_tokens = self.count_tokens(messages)
        if prompt_tokens > self.max_input_tokens:
            raise GenerationError(f"prompt has {prompt_tokens} tokens > max_input_tokens={self.max_input_tokens}")
        words = self.responder(messages).split()
        finish = "length" if len(words) > params.max_new_tokens else "stop"
        words = words[: params.max_new_tokens]
        return GenerationResult(
            text=" ".join(words),
            model_version=self.model_version,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(words),
            latency_ms=self.latency_ms,
            finish_reason=finish,
        )

    def generate_batch(self, batch: list[list[ChatMessage]], params: GenerationParams) -> list[GenerationResult]:
        return [self.generate(m, params) for m in batch]
