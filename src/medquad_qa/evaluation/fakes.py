"""Deterministic test doubles for evaluation, security and integration tests (synthetic text only)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from medquad_qa.contracts import ChatMessage, GenerationParams, GenerationResult


class ScriptedGenerator:
    """Generator that returns a fixed text (or a function of the messages) and records every prompt.

    ``raise_exc`` makes ``generate`` raise; ``delay_s`` simulates slow generation.
    """

    def __init__(
        self,
        text: str | Callable[[list[ChatMessage]], str] = "Synthetic answer [E1].",
        *,
        model_version: str = "scripted-generator@0",
        raise_exc: BaseException | None = None,
        delay_s: float = 0.0,
    ) -> None:
        self.text = text
        self.model_version = model_version
        self.raise_exc = raise_exc
        self.delay_s = delay_s
        self.prompts: list[list[ChatMessage]] = []
        self._lock = threading.Lock()

    def generate(self, messages: list[ChatMessage], params: GenerationParams) -> GenerationResult:
        with self._lock:
            self.prompts.append(list(messages))
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.raise_exc is not None:
            raise self.raise_exc
        out = self.text(messages) if callable(self.text) else self.text
        return GenerationResult(
            text=out,
            model_version=self.model_version,
            prompt_tokens=sum(len(m.content.split()) for m in messages),
            completion_tokens=len(out.split()),
            latency_ms=1.0,
        )
