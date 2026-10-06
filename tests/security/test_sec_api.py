"""Adversarial tests of the HTTP service (E5): input bounds, error leakage, request IDs, logs, metrics.

The API is exercised in-process with the real ``RagPipeline`` (fixture retriever + scripted generator), so
these tests also cover the API <-> pipeline error mapping. A canary string is placed in every request; it
must never appear in error bodies, logs or /metrics.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from medquad_qa.api.app import create_app
from medquad_qa.api.settings import ServiceSettings
from medquad_qa.contracts import (
    REQUEST_ID_PATTERN,
    ArtifactUnavailableError,
    GenerationError,
    GenerationTimeoutError,
)
from medquad_qa.evaluation.fakes import ScriptedGenerator
from medquad_qa.evaluation.fixture_retriever import FixtureRetriever
from medquad_qa.observability.logs import JsonFormatter
from medquad_qa.rag.pipeline import RagPipeline

CANARY = "ZQXCANARY7731"
Q = f"What causes synthetic condition {CANARY}?"
EVIDENCE = [f"Synthetic condition {CANARY} is caused by vitamin Q deficiency."]


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.setFormatter(JsonFormatter())
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


@pytest.fixture
def logs() -> Iterator[_Capture]:
    h = _Capture()
    root = logging.getLogger()
    old = root.level
    root.addHandler(h)
    root.setLevel(logging.DEBUG)
    yield h
    root.removeHandler(h)
    root.setLevel(old)


def make(gen: ScriptedGenerator, **settings: Any) -> TestClient:
    def provider(observer: Any) -> RagPipeline:
        pipe = RagPipeline(
            retriever=FixtureRetriever({Q: EVIDENCE}), generator_provider=lambda _v: gen, observer=observer
        )
        pipe.preload(("base",))
        return pipe

    app = create_app(ServiceSettings(build_in_background=False, **settings), provider)
    client = TestClient(app, raise_server_exceptions=False)
    client.__enter__()
    return client


@pytest.fixture
def client() -> Iterator[TestClient]:
    c = make(ScriptedGenerator(f"{CANARY} is caused by vitamin Q deficiency [E1]."))
    yield c
    c.__exit__(None, None, None)


def no_canary(text: str) -> bool:
    return CANARY not in text


# ------------------------------------------------------------------ T6 input bounds
def test_body_limit_413_with_and_without_content_length(client: TestClient) -> None:
    big = json.dumps({"question": "x" * 20_000}).encode()
    r = client.post("/v1/qa", content=big, headers={"content-type": "application/json"})
    assert r.status_code == 413
    r = client.post("/v1/qa", content=iter([big[:8000], big[8000:]]), headers={"content-type": "application/json"})
    assert r.status_code == 413


@pytest.mark.parametrize(
    "body",
    [
        {"question": f"{CANARY}\x00bad"},
        {"question": "x" * 2001 + CANARY},
        {"question": Q, "top_k": 999},
        {"question": Q, "experiment_mode": f"clinical_{CANARY}"},
        {"question": Q, "generation": {"max_new_tokens": 10**6}},
        {"question": Q, "generation": {"temperature": -1}},
        {"question": [CANARY]},
    ],
)
def test_validation_errors_never_echo_input(client: TestClient, body: dict[str, Any]) -> None:
    r = client.post("/v1/qa", json=body)
    assert r.status_code == 422
    assert r.json()["error_code"] == "validation_error"
    assert no_canary(r.text), r.text


def test_validation_error_does_not_echo_unknown_key(client: TestClient) -> None:
    """F-005 regression (fixed by service-platform-engineer; retest PASS)."""
    r = client.post("/v1/qa", json={"question": Q, CANARY: 1})
    assert no_canary(r.text)


def test_malformed_json_422(client: TestClient) -> None:
    r = client.post(
        "/v1/qa", content=b'{"question": "' + CANARY.encode() + b'"', headers={"content-type": "application/json"}
    )
    assert r.status_code == 422 and no_canary(r.text)


# ------------------------------------------------------------------ T7 request IDs
@pytest.mark.parametrize("rid", ["a\r\nX-Injected: 1", "x" * 65, "ü-unicode", "../../etc", ""])
def test_bad_request_ids_replaced(client: TestClient, rid: str) -> None:
    r = client.get("/health/live", headers=[(b"x-request-id", rid.encode("utf-8"))])
    out = r.headers["x-request-id"]
    assert out != rid and re.fullmatch(REQUEST_ID_PATTERN, out)
    assert "x-injected" not in {k.lower() for k in r.headers}


def test_good_request_id_echoed(client: TestClient) -> None:
    r = client.post("/v1/qa", json={"question": Q}, headers={"X-Request-ID": "eval.run-1_A"})
    assert (
        r.status_code == 200
        and r.headers["x-request-id"] == "eval.run-1_A"
        and r.json()["request_id"] == "eval.run-1_A"
    )


# ------------------------------------------------------------------ T8/T9/T13 error mapping and leakage
@pytest.mark.parametrize(
    "exc,status,code",
    [
        (ArtifactUnavailableError(f"missing /secret/path/{CANARY}"), 503, "artifact_unavailable"),
        (GenerationTimeoutError(f"slow {CANARY}"), 504, "generation_timeout"),
        (GenerationError(f"boom {CANARY}"), 502, "generation_failed"),
        (RuntimeError(f"Traceback /home/user/{CANARY}"), 500, "internal_error"),
    ],
)
def test_pipeline_errors_mapped_without_leaks(logs: _Capture, exc: Exception, status: int, code: str) -> None:
    c = make(ScriptedGenerator(raise_exc=exc))
    try:
        r = c.post("/v1/qa", json={"question": Q})
    finally:
        c.__exit__(None, None, None)
    assert r.status_code == status and r.json()["error_code"] == code
    assert no_canary(r.text) and "Traceback" not in r.text and "/home/" not in r.text
    if status == 503:
        assert r.headers.get("retry-after")
    assert all(no_canary(line) for line in logs.lines), [ln for ln in logs.lines if CANARY in ln]


def test_api_backstop_timeout_504() -> None:
    c = make(ScriptedGenerator("late [E1].", delay_s=1.0), request_timeout_s=0.2)
    try:
        r = c.post("/v1/qa", json={"question": Q})
    finally:
        c.__exit__(None, None, None)
    assert r.status_code == 504 and r.json()["error_code"] == "request_timeout"


def test_unavailable_mode_503(client: TestClient) -> None:
    r = client.post("/v1/qa", json={"question": Q, "experiment_mode": "finetuned"})
    assert r.status_code == 503 and r.json()["error_code"] == "mode_unavailable" and r.headers.get("retry-after")


# ------------------------------------------------------------------ T11 logging and metrics privacy
def test_no_content_in_logs_or_metrics_by_default(client: TestClient, logs: _Capture) -> None:
    r = client.post("/v1/qa", json={"question": Q})
    assert r.status_code == 200 and not r.json()["abstained"]
    assert any('"qa_request"' in ln for ln in logs.lines)
    assert all(no_canary(ln) for ln in logs.lines), [ln for ln in logs.lines if CANARY in ln]
    assert no_canary(client.get("/metrics").text)


def test_opt_in_content_logging_is_bounded(logs: _Capture) -> None:
    c = make(ScriptedGenerator(("A" * 500) + f" {CANARY} [E1]."), log_content=True)
    try:
        c.post("/v1/qa", json={"question": Q})
    finally:
        c.__exit__(None, None, None)
    qa = [json.loads(ln) for ln in logs.lines if '"qa_request"' in ln]
    assert qa and len(qa[-1]["answer_preview"]) <= 200 and CANARY not in qa[-1]["answer_preview"]
    assert any("content_logging_enabled" in ln for ln in logs.lines)
