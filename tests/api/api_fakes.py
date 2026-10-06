"""Offline API test fixtures: a synthetic FakePipeline (no models, no data, no network).

All texts and record IDs here are synthetic and are not MedQuAD content.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from medquad_qa.api.app import create_app
from medquad_qa.api.settings import ServiceSettings
from medquad_qa.contracts import (
    EXPERIMENT_MODES,
    RAG_MODES,
    Citation,
    ComponentStatus,
    ExperimentMode,
    GenerationMeta,
    GenerationParams,
    PipelineObserver,
    PredictorOutput,
    QARequest,
    QAResponse,
    RetrievalHit,
)

SYNTH_IDS = ["mq-0000000000000001", "mq-0000000000000002"]


class FakePipeline:
    """Implements QAPipeline + ReadinessReporter structurally. Behaviour is set per test."""

    def __init__(
        self,
        observer: PipelineObserver | None = None,
        *,
        raise_exc: BaseException | None = None,
        delay_s: float = 0.0,
        abstain: str | None = None,
        modes: frozenset[ExperimentMode] = frozenset(EXPERIMENT_MODES),
        components: list[ComponentStatus] | None = None,
        invalid_ids: list[str] | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        self.observer = observer
        self.raise_exc = raise_exc
        self.delay_s = delay_s
        self.abstain = abstain
        self.modes = modes
        self.invalid_ids = invalid_ids or []
        self.warnings = warnings or []
        self._components = components
        self.calls = 0
        self.max_concurrent = 0
        self._active = 0
        self._lock = threading.Lock()
        self.release = threading.Event()
        self.release.set()

    def readiness(self) -> list[ComponentStatus]:
        if self._components is not None:
            return self._components
        return [
            ComponentStatus(name="corpus", ok=True, required=True, version="synthetic-corpus-0"),
            ComponentStatus(name="retriever:bm25", ok=True, required=True, version="synthetic-index-0"),
            ComponentStatus(name="generator:base", ok=True, required=True, version="fake-gen@0"),
        ]

    def available_modes(self) -> frozenset[ExperimentMode]:
        return self.modes

    def versions(self) -> dict[str, str | None]:
        return {
            "model_version": "fake-gen@0",
            "finetuned_model_version": None,
            "corpus_version": "synthetic-corpus-0",
            "index_version": "synthetic-index-0",
            "prompt_version": "rag-v1+00000000",
        }

    def answer(self, request: QARequest, request_id: str) -> QAResponse:
        with self._lock:
            self.calls += 1
            self._active += 1
            self.max_concurrent = max(self.max_concurrent, self._active)
        try:
            if self.delay_s:
                time.sleep(self.delay_s)
            self.release.wait(timeout=30)
            if self.raise_exc is not None:
                raise self.raise_exc
            return self._response(request, request_id)
        finally:
            with self._lock:
                self._active -= 1

    def _response(self, request: QARequest, request_id: str) -> QAResponse:
        mode = request.experiment_mode
        rag = mode in RAG_MODES
        hits = (
            [
                RetrievalHit(
                    record_id=rid,
                    rank=i + 1,
                    score=0.03 - 0.01 * i,
                    retriever="hybrid_rrf:qa",
                    evidence_text="Synthetic evidence text.",
                    corpus_version="synthetic-corpus-0",
                )
                for i, rid in enumerate(SYNTH_IDS)
            ]
            if rag
            else []
        )
        if self.observer is not None and rag:
            self.observer.on_retrieval(mode, hits)
            self.observer.on_gate(
                mode,
                PredictorOutput(
                    answerability_score=0.8, predicted_label=True, threshold_version="t0", model_version="fake-clf"
                ),
                "predictor",
            )
        common: dict[str, Any] = {
            "request_id": request_id,
            "experiment_mode": mode,
            "retrieved_record_ids": [h.record_id for h in hits],
            "model_version": "fake-gen@0",
            "corpus_version": "synthetic-corpus-0" if rag else None,
            "prompt_version": "rag-v1+00000000",
            "latency_ms": 12.5,
            "component_latency_ms": {"retrieval": 3.0, "generation": 8.0} if rag else {"generation": 8.0},
            "warnings": self.warnings,
            "invalid_citation_ids": self.invalid_ids,
            "retriever": "hybrid_rrf:qa" if rag else None,
        }
        if self.abstain:
            return QAResponse(
                answer="I cannot answer this from the available evidence.",
                abstained=True,
                abstention_reason=self.abstain,
                **common,
            )
        cites = [Citation(record_id=SYNTH_IDS[0], evidence_snippet="Synthetic evidence text.")] if rag else []
        return QAResponse(
            answer=f"Synthetic answer [{SYNTH_IDS[0]}]." if rag else "Synthetic answer.",
            abstained=False,
            citations=cites,
            generation=GenerationMeta(
                params=request.generation or GenerationParams(),
                prompt_tokens=120,
                completion_tokens=30,
                finish_reason="stop",
            ),
            **common,
        )


ClientFactory = Callable[..., tuple[TestClient, FakePipeline]]


def wait_built(client: TestClient) -> None:
    assert client.app.state.pipeline.built.wait(5), "pipeline build did not finish"  # type: ignore[attr-defined]


@pytest.fixture
def make_client() -> Iterator[ClientFactory]:
    clients: list[TestClient] = []

    def _make(settings: ServiceSettings | None = None, **fake_kwargs: Any) -> tuple[TestClient, FakePipeline]:
        holder: dict[str, FakePipeline] = {}

        def provider(observer: PipelineObserver | None) -> FakePipeline:
            holder["p"] = FakePipeline(observer, **fake_kwargs)
            return holder["p"]

        app = create_app(settings or ServiceSettings(), provider)
        client = TestClient(app, raise_server_exceptions=False)
        client.__enter__()
        clients.append(client)
        wait_built(client)
        return client, holder["p"]

    yield _make
    for c in clients:
        c.__exit__(None, None, None)
