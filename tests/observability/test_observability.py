"""Offline tests for metrics, structured logging and the offline-eval exporter."""

from __future__ import annotations

import io
import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from api_fakes import ClientFactory
from prometheus_client.parser import text_string_to_metric_families

from medquad_qa.api.settings import ServiceSettings
from medquad_qa.contracts import RetrievalHit
from medquad_qa.observability.logs import JsonFormatter, configure_logging
from medquad_qa.observability.metrics import ServiceMetrics
from medquad_qa.observability.observer import MetricsObserver, retriever_label
from medquad_qa.observability.offline_eval import OfflineEvalExporter

QUESTION = "What causes synthetic marker condition ZQX-4417?"
Q = {"question": QUESTION, "experiment_mode": "rag", "top_k": 3}


def samples(text: str) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    out = {}
    for fam in text_string_to_metric_families(text):
        for s in fam.samples:
            out[(s.name, tuple(sorted(s.labels.items())))] = s.value
    return out


def value(text: str, name: str, **labels: str) -> float:
    return samples(text).get((name, tuple(sorted(labels.items()))), 0.0)


@pytest.fixture
def log_stream() -> Iterator[io.StringIO]:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    lg = logging.getLogger("medquad_qa")
    old_level = lg.level
    lg.addHandler(handler)
    lg.setLevel(logging.INFO)
    yield stream
    lg.removeHandler(handler)
    lg.setLevel(old_level)


def log_lines(stream: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


# ---------------------------------------------------------------- metrics endpoint
def test_metrics_catalogue_after_requests(make_client: ClientFactory) -> None:
    client, _ = make_client()
    assert client.post("/v1/qa", json=Q).status_code == 200
    assert client.post("/v1/qa", json={**Q, "experiment_mode": "base"}).status_code == 200
    assert client.post("/v1/qa", json={"question": "x"}).status_code == 422
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    t = r.text
    assert value(t, "medquad_qa_requests_total", mode="rag", outcome="answered") == 1
    assert value(t, "medquad_qa_requests_total", mode="base", outcome="answered") == 1
    assert value(t, "medquad_http_requests_total", method="POST", route="/v1/qa", status="422") == 1
    assert value(t, "medquad_qa_latency_seconds_count", mode="rag") == 1
    assert value(t, "medquad_qa_component_latency_seconds_count", mode="rag", component="retrieval") == 1
    assert value(t, "medquad_qa_input_chars_count", mode="rag") == 1
    assert value(t, "medquad_retrieval_top_score_count", retriever="hybrid_rrf:qa") == 1
    assert value(t, "medquad_retrieval_score_count", retriever="hybrid_rrf:qa") == 2
    assert value(t, "medquad_answerability_score_count", gate="predictor") == 1
    assert value(t, "medquad_generation_prompt_tokens_count", mode="rag") == 1
    assert value(t, "medquad_pipeline_state", state="ready") == 1
    assert value(t, "medquad_artifact_info", artifact="index_version", version="synthetic-index-0") == 1
    assert value(t, "medquad_component_ready", component="corpus", required="true") == 1
    assert value(t, "medquad_mode_available", mode="finetuned_rag") == 1
    assert "contracts_version" in t
    # privacy: no question text and no record IDs as label values
    assert "ZQX-4417" not in t
    assert "mq-0000000000000001" not in t


def test_abstention_and_citation_failure_metrics(make_client: ClientFactory) -> None:
    client, _ = make_client(abstain="invalid_citations", invalid_ids=["E9", "mq-ffffffffffffffff"])
    client.post("/v1/qa", json=Q)
    t = client.get("/metrics").text
    assert value(t, "medquad_qa_requests_total", mode="rag", outcome="abstained") == 1
    assert value(t, "medquad_qa_abstentions_total", mode="rag", reason="invalid_citations") == 1
    assert value(t, "medquad_qa_citation_failures_total", mode="rag", kind="invalid_citations") == 1
    assert value(t, "medquad_qa_citation_failures_total", mode="rag", kind="invalid_ids") == 1
    assert value(t, "medquad_qa_invalid_citation_ids_total", mode="rag") == 2
    assert "mq-ffffffffffffffff" not in t


def test_warning_labels_are_bounded(make_client: ClientFactory) -> None:
    client, _ = make_client(warnings=["lexical_fallback", "free text " + QUESTION])
    client.post("/v1/qa", json=Q)
    t = client.get("/metrics").text
    assert value(t, "medquad_qa_warnings_total", mode="rag", warning="lexical_fallback") == 1
    assert value(t, "medquad_qa_warnings_total", mode="rag", warning="other") == 1
    assert "ZQX-4417" not in t


def test_evidence_filtered_warning_and_version_are_kept(make_client: ClientFactory) -> None:
    """D-060: the `evidence_filtered` warning and `evidence_filter_version` key pass through unchanged."""
    client, fake = make_client(warnings=["evidence_filtered"])
    base_versions = fake.versions()
    fake.versions = lambda: {**base_versions, "evidence_filter_version": "evf-v1+0000abcd"}
    r = client.post("/v1/qa", json=Q)
    assert r.status_code == 200
    assert r.json()["warnings"] == ["evidence_filtered"]
    assert client.get("/v1/info").json()["versions"]["evidence_filter_version"] == "evf-v1+0000abcd"
    t = client.get("/metrics").text
    assert value(t, "medquad_qa_warnings_total", mode="rag", warning="evidence_filtered") == 1
    assert value(t, "medquad_qa_warnings_total", mode="rag", warning="other") == 0
    assert value(t, "medquad_artifact_info", artifact="evidence_filter_version", version="evf-v1+0000abcd") == 1


@pytest.mark.parametrize("mode", ["rag", "base"])
def test_safety_check_abstention_is_counted_not_an_error(make_client: ClientFactory, mode: str) -> None:
    """safety-v4 (F-009): a model-check refusal or failure is a normal abstention with a known warning label,
    and `safety_check_model` passes through to /v1/info and artifact_info."""
    client, fake = make_client(abstain="personalized_medical_advice", warnings=["safety_check_failed"])
    base_versions = fake.versions()
    fake.versions = lambda: {**base_versions, "safety_check_model": "synthetic-safety-clf@0"}
    r = client.post("/v1/qa", json={**Q, "experiment_mode": mode})
    assert r.status_code == 200
    body = r.json()
    assert body["abstained"] is True and body["abstention_reason"] == "personalized_medical_advice"
    assert body["warnings"] == ["safety_check_failed"]
    assert client.get("/v1/info").json()["versions"]["safety_check_model"] == "synthetic-safety-clf@0"
    t = client.get("/metrics").text
    assert value(t, "medquad_qa_requests_total", mode=mode, outcome="abstained") == 1
    assert value(t, "medquad_qa_requests_total", mode=mode, outcome="error") == 0
    assert value(t, "medquad_qa_abstentions_total", mode=mode, reason="personalized_medical_advice") == 1
    assert value(t, "medquad_qa_abstentions_total", mode=mode, reason="other") == 0
    assert value(t, "medquad_qa_warnings_total", mode=mode, warning="safety_check_failed") == 1
    assert value(t, "medquad_qa_warnings_total", mode=mode, warning="other") == 0
    assert value(t, "medquad_artifact_info", artifact="safety_check_model", version="synthetic-safety-clf@0") == 1


def test_error_metrics(make_client: ClientFactory) -> None:
    from medquad_qa.contracts import GenerationTimeoutError

    client, _ = make_client(raise_exc=GenerationTimeoutError("t"))
    client.post("/v1/qa", json=Q)
    t = client.get("/metrics").text
    assert value(t, "medquad_qa_requests_total", mode="rag", outcome="error") == 1
    assert value(t, "medquad_qa_errors_total", mode="rag", error_code="generation_timeout") == 1


def test_unmatched_routes_do_not_create_label_cardinality(make_client: ClientFactory) -> None:
    client, _ = make_client()
    for i in range(5):
        client.get(f"/random/{i}")
    t = client.get("/metrics").text
    assert value(t, "medquad_http_requests_total", method="GET", route="unmatched", status="404") == 5
    assert "/random/" not in t


def test_failed_build_metrics() -> None:
    from fastapi.testclient import TestClient

    from medquad_qa.api.app import create_app

    def provider(observer: object) -> object:
        raise FileNotFoundError("no index")

    with TestClient(create_app(ServiceSettings(), provider)) as client:
        client.app.state.pipeline.built.wait(5)  # type: ignore[attr-defined]
        t = client.get("/metrics").text
    assert value(t, "medquad_pipeline_state", state="failed") == 1
    assert value(t, "medquad_pipeline_state", state="ready") == 0


# ---------------------------------------------------------------- observer
def test_observer_swallows_errors_and_bounds_labels() -> None:
    m = ServiceMetrics()
    obs = MetricsObserver(m)
    hit = RetrievalHit(
        record_id="mq-0000000000000001",
        rank=1,
        score=7.5,
        retriever="weird-retriever",
        evidence_text="synthetic",
        corpus_version="c0",
    )
    obs.on_retrieval("rag", [hit])
    obs.on_gate("rag", None, "nonsense-gate")
    broken: Any = None
    obs.on_retrieval("rag", broken)
    from prometheus_client import generate_latest

    t = generate_latest(m.registry).decode()
    assert value(t, "medquad_retrieval_top_score_count", retriever="other") == 1
    assert value(t, "medquad_answerability_gate_decisions_total", gate="other", label="none") == 1
    assert value(t, "medquad_observer_errors_total", hook="on_retrieval") == 1


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("bm25:answer", "bm25:answer"),
        ("hybrid_rrf+ce:qa", "hybrid_rrf+ce:qa"),
        ("dense:xx", "dense"),
        ("x:qa", "other"),
    ],
)
def test_retriever_label(name: str, label: str) -> None:
    assert retriever_label(name) == label


# ---------------------------------------------------------------- logging privacy
def test_logs_have_no_content_by_default(make_client: ClientFactory, log_stream: io.StringIO) -> None:
    client, _ = make_client()
    client.post("/v1/qa", json=Q, headers={"X-Request-ID": "log-test-1"})
    lines = log_lines(log_stream)
    qa = [ln for ln in lines if ln["msg"] == "qa_request"]
    access = [ln for ln in lines if ln["msg"] == "http_request"]
    assert len(qa) == 1 and len(access) == 1
    assert qa[0]["request_id"] == "log-test-1" and access[0]["request_id"] == "log-test-1"
    assert qa[0]["mode"] == "rag" and qa[0]["outcome"] == "answered"
    assert "question_sha256" not in qa[0]
    raw = log_stream.getvalue()
    assert "ZQX-4417" not in raw
    assert "Synthetic answer" not in raw


def test_validation_error_does_not_log_input(make_client: ClientFactory, log_stream: io.StringIO) -> None:
    client, _ = make_client()
    client.post("/v1/qa", json={"question": QUESTION, "top_k": 999})
    assert "ZQX-4417" not in log_stream.getvalue()


def test_internal_error_logs_type_only(make_client: ClientFactory, log_stream: io.StringIO) -> None:
    client, _ = make_client(raise_exc=RuntimeError("leak " + QUESTION))
    r = client.post("/v1/qa", json=Q)
    assert r.status_code == 500
    raw = log_stream.getvalue()
    assert "ZQX-4417" not in raw and "ZQX-4417" not in r.text
    assert '"error_type": "RuntimeError"' in raw


def test_content_logging_opt_in_is_minimised(make_client: ClientFactory, log_stream: io.StringIO) -> None:
    long_q = "Synthetic opt-in question " + "z" * 400
    client, _ = make_client(ServiceSettings(log_content=True))
    client.post("/v1/qa", json={**Q, "question": long_q})
    lines = log_lines(log_stream)
    assert any(ln["msg"] == "content_logging_enabled" and ln["level"] == "WARNING" for ln in lines)
    qa = next(ln for ln in lines if ln["msg"] == "qa_request")
    assert len(str(qa["question_preview"])) == 200
    assert len(str(qa["question_sha256"])) == 64
    assert long_q not in log_stream.getvalue()


def test_json_formatter_exception_type_only() -> None:
    fmt = JsonFormatter()
    try:
        raise ValueError("sensitive " + QUESTION)
    except ValueError:
        import sys

        rec = logging.LogRecord("x", logging.ERROR, __file__, 1, "failed", (), sys.exc_info())
    out = json.loads(fmt.format(rec))
    assert out["exc_type"] == "ValueError"
    assert "ZQX-4417" not in json.dumps(out)


def test_configure_logging_installs_single_json_handler() -> None:
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    try:
        configure_logging("WARNING")
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JsonFormatter)
        assert root.level == logging.WARNING
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


# ---------------------------------------------------------------- offline eval export
def _write_eval(path: Path, run_id: str, val: float) -> None:
    path.write_text(
        json.dumps(
            {
                "eval_run_id": run_id,
                "completed_at": "2026-10-07T12:00:00+00:00",
                "metrics": [{"track": "B", "mode": "rag", "metric": "citation_validity", "value": val}],
            }
        )
    )


def test_offline_eval_exporter_reload_and_invalid(tmp_path: Path) -> None:
    m = ServiceMetrics()
    p = tmp_path / "summary.json"
    exp = OfflineEvalExporter(m, str(p))
    exp.refresh()  # missing file: no error
    assert exp.loaded_run_id is None
    _write_eval(p, "e4-run1", 0.9)
    exp.refresh()
    assert exp.loaded_run_id == "e4-run1"
    _write_eval(p, "e4-run2", 0.95)
    os.utime(p, (p.stat().st_atime, p.stat().st_mtime + 10))
    exp.refresh()
    from prometheus_client import generate_latest

    t = generate_latest(m.registry).decode()
    assert (
        value(
            t, "medquad_offline_eval_metric", eval_run_id="e4-run2", track="B", mode="rag", metric="citation_validity"
        )
        == 0.95
    )
    assert "e4-run1" not in t
    p.write_text('{"eval_run_id": "bad id with spaces", "metrics": []}')
    os.utime(p, (p.stat().st_atime, p.stat().st_mtime + 20))
    exp.refresh()
    assert exp.loaded_run_id is None
    assert "medquad_offline_eval_metric{" not in generate_latest(m.registry).decode()


def test_offline_eval_on_metrics_endpoint(make_client: ClientFactory, tmp_path: Path) -> None:
    p = tmp_path / "summary.json"
    _write_eval(p, "e4-run9", 0.8)
    client, _ = make_client(ServiceSettings(offline_eval_path=str(p)))
    t = client.get("/metrics").text
    assert (
        value(
            t, "medquad_offline_eval_metric", eval_run_id="e4-run9", track="B", mode="rag", metric="citation_validity"
        )
        == 0.8
    )
    assert value(t, "medquad_offline_eval_timestamp_seconds", eval_run_id="e4-run9") > 0
