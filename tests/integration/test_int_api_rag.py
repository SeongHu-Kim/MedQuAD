"""Integration: real API app + real RagPipeline (fixture retriever, scripted generator), all four modes."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from medquad_qa.api.app import create_app
from medquad_qa.api.settings import ServiceSettings
from medquad_qa.contracts import QAResponse
from medquad_qa.evaluation.fakes import ScriptedGenerator
from medquad_qa.evaluation.fixture_retriever import FixtureRetriever, fixture_record_id
from medquad_qa.evaluation.qa_report import summarize_mode
from medquad_qa.rag.pipeline import RagPipeline

pytestmark = pytest.mark.integration

Q = "What causes synthetic condition X?"
EV = ["Synthetic condition X is caused by vitamin Q deficiency.", "Synthetic condition X is not contagious."]


@pytest.fixture
def client() -> Iterator[TestClient]:
    gens = {
        v: ScriptedGenerator("X is caused by vitamin Q deficiency [E1].", model_version=f"{v}@0")
        for v in ("base", "finetuned")
    }

    def provider(observer: Any) -> RagPipeline:
        pipe = RagPipeline(retriever=FixtureRetriever({Q: EV}), generator_provider=lambda v: gens[v], observer=observer)
        pipe.preload()
        return pipe

    c = TestClient(create_app(ServiceSettings(build_in_background=False), provider), raise_server_exceptions=False)
    c.__enter__()
    yield c
    c.__exit__(None, None, None)


@pytest.mark.parametrize("mode", ["base", "rag", "finetuned", "finetuned_rag"])
def test_all_modes_end_to_end(client: TestClient, mode: str) -> None:
    r = client.post("/v1/qa", json={"question": Q, "experiment_mode": mode, "top_k": 2})
    assert r.status_code == 200
    resp = QAResponse.model_validate(r.json())
    assert resp.experiment_mode == mode and not resp.abstained
    assert resp.model_version.startswith("finetuned" if "finetuned" in mode else "base")
    if "rag" in mode:
        assert [c.record_id for c in resp.citations] == [fixture_record_id(EV[0])]
        assert {c.record_id for c in resp.citations} <= set(resp.retrieved_record_ids)
        assert f"[{fixture_record_id(EV[0])}]" in resp.answer
        assert set(resp.component_latency_ms) >= {"safety_rules", "retrieval", "generation", "citation_validation"}
    else:
        assert resp.citations == [] and resp.retrieved_record_ids == []


def test_no_hits_abstains_and_metrics_exported(client: TestClient) -> None:
    r = client.post("/v1/qa", json={"question": "What is unrelated condition Z?", "experiment_mode": "rag"})
    assert r.status_code == 200 and r.json()["abstention_reason"] == "no_relevant_evidence"
    metrics = client.get("/metrics").text
    assert "medquad_" in metrics


def test_harness_consumes_api_responses(client: TestClient) -> None:
    from medquad_qa.contracts import EvaluationExample

    ex = EvaluationExample(
        example_id="int-1",
        question=Q,
        gold_record_ids=[fixture_record_id(EV[0])],
        answerable=True,
        label_provenance="synthetic_rule",
        question_provenance="synthetic_rule",
        eval_split="dev",
        evaluation_track="corpus_grounded",
    )
    resp = QAResponse.model_validate(client.post("/v1/qa", json={"question": Q, "top_k": 2}).json())
    s = summarize_mode([ex], [resp])
    assert s["supplied_evidence_retrieval"]["recall@1"] == 1.0 and s["abstention"]["all"]["tn"] == 1
