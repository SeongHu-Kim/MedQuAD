"""Offline API tests against a synthetic FakePipeline."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import pytest
from api_fakes import ClientFactory, FakePipeline, wait_built
from fastapi.testclient import TestClient

from medquad_qa.api.app import create_app
from medquad_qa.api.settings import ServiceSettings
from medquad_qa.contracts import (
    ArtifactUnavailableError,
    ComponentStatus,
    ContractViolationError,
    GenerationError,
    GenerationTimeoutError,
    ModeUnavailableError,
    QAResponse,
    RetrieverUnavailableError,
)

Q = {"question": "What are the symptoms of synthetic condition X?", "experiment_mode": "rag", "top_k": 3}


# ---------------------------------------------------------------- health / readiness
def test_live_and_ready(make_client: ClientFactory) -> None:
    client, _ = make_client()
    assert client.get("/health/live").json() == {"status": "alive"}
    r = client.get("/health/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["available_modes"] == ["base", "finetuned", "finetuned_rag", "rag"]
    assert {c["name"] for c in body["components"]} == {"corpus", "retriever:bm25", "generator:base"}


def test_build_failure_keeps_live_and_reports_not_ready() -> None:
    def provider(observer: object) -> object:
        raise ArtifactUnavailableError("index manifest missing: artifacts/indexes/manifests/x.json")

    with TestClient(create_app(ServiceSettings(), provider)) as client:
        wait_built(client)
        assert client.get("/health/live").status_code == 200
        r = client.get("/health/ready")
        assert r.status_code == 503
        assert r.headers["retry-after"] == "5"
        assert r.json()["pipeline"] == "failed"
        assert "ArtifactUnavailableError" in r.json()["failure"]
        q = client.post("/v1/qa", json=Q)
        assert q.status_code == 503
        assert q.json()["error_code"] == "not_ready"


def test_missing_rag_package_is_a_readiness_failure_not_a_crash() -> None:
    def provider(observer: object) -> object:
        raise ImportError("No module named 'medquad_qa.rag.factory'")

    with TestClient(create_app(ServiceSettings(), provider)) as client:
        wait_built(client)
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503


def test_ready_while_building_is_503() -> None:
    gate = threading.Event()

    def provider(observer: object) -> FakePipeline:
        gate.wait(5)
        return FakePipeline()

    with TestClient(create_app(ServiceSettings(), provider)) as client:
        assert client.get("/health/live").status_code == 200
        r = client.get("/health/ready")
        assert r.status_code == 503
        assert r.json()["status"] == "starting"
        gate.set()
        wait_built(client)
        assert client.get("/health/ready").status_code == 200


def test_required_component_down_is_not_ready(make_client: ClientFactory) -> None:
    comps = [
        ComponentStatus(name="corpus", ok=True, required=True),
        ComponentStatus(name="retriever:dense", ok=False, required=True, detail="qdrant unreachable"),
    ]
    client, _ = make_client(components=comps)
    r = client.get("/health/ready")
    assert r.status_code == 503
    assert r.json()["status"] == "not_ready"


def test_optional_component_down_is_degraded_but_ready(make_client: ClientFactory) -> None:
    comps = [
        ComponentStatus(name="corpus", ok=True, required=True),
        ComponentStatus(name="generator:finetuned", ok=False, required=False, detail="adapter missing"),
    ]
    client, _ = make_client(components=comps, modes=frozenset({"base", "rag"}))
    r = client.get("/health/ready")
    assert r.status_code == 200
    assert r.json()["degraded"] is True


# ---------------------------------------------------------------- QA happy paths
@pytest.mark.parametrize("mode", ["base", "rag", "finetuned", "finetuned_rag"])
def test_qa_modes_are_explicit(make_client: ClientFactory, mode: str) -> None:
    client, _ = make_client()
    r = client.post("/v1/qa", json={**Q, "experiment_mode": mode})
    assert r.status_code == 200, r.text
    body = QAResponse.model_validate(r.json())
    assert body.experiment_mode == mode
    if mode in ("base", "finetuned"):
        assert body.citations == [] and body.retrieved_record_ids == []
    else:
        assert {c.record_id for c in body.citations} <= set(body.retrieved_record_ids)
    assert "Not medical advice" in body.disclaimer


def test_request_id_propagation(make_client: ClientFactory) -> None:
    client, _ = make_client()
    r = client.post("/v1/qa", json=Q, headers={"X-Request-ID": "abc-123.x_y"})
    assert r.headers["x-request-id"] == "abc-123.x_y"
    assert r.json()["request_id"] == "abc-123.x_y"


@pytest.mark.parametrize("bad", ["has space", "x" * 65, "<script>", "aéb", ""])
def test_request_id_sanitised(make_client: ClientFactory, bad: str) -> None:
    client, _ = make_client()
    r = client.post("/v1/qa", json=Q, headers=[(b"x-request-id", bad.encode("utf-8"))])
    rid = r.headers["x-request-id"]
    assert rid != bad and len(rid) == 32
    assert r.json()["request_id"] == rid


def test_request_id_on_errors_and_health(make_client: ClientFactory) -> None:
    client, _ = make_client()
    assert client.get("/health/live", headers={"X-Request-ID": "h1"}).headers["x-request-id"] == "h1"
    r = client.post("/v1/qa", json={"question": "x"}, headers={"X-Request-ID": "v1"})
    assert r.status_code == 422
    assert r.headers["x-request-id"] == "v1" and r.json()["request_id"] == "v1"


def test_info_endpoint(make_client: ClientFactory) -> None:
    client, _ = make_client()
    body = client.get("/v1/info").json()
    assert body["versions"]["index_version"] == "synthetic-index-0"
    assert body["limits"]["max_question_chars"] == 2000
    assert body["contracts_version"]


# ---------------------------------------------------------------- input bounds
SECRET = "SYNTHETIC-SECRET-QUESTION-TEXT"  # noqa: S105 - marker text, not a credential


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "hi"},
        {"question": SECRET + "x" * 2000},
        {"question": SECRET + "\x00bad"},
        {"question": SECRET, "experiment_mode": "gpt"},
        {"question": SECRET, "top_k": 0},
        {"question": SECRET, "top_k": 21},
        {"question": SECRET, "generation": {"max_new_tokens": 5000}},
        {"question": SECRET, "generation": {"temperature": 3}},
        {"question": SECRET, "unexpected": SECRET},
        {"question": 12345},
    ],
)
def test_validation_422_never_echoes_input(make_client: ClientFactory, payload: dict[str, object]) -> None:
    client, fake = make_client()
    r = client.post("/v1/qa", json=payload)
    assert r.status_code == 422
    body = r.json()
    assert body["error_code"] == "validation_error"
    assert SECRET not in r.text and "12345" not in r.text
    assert all(set(e) == {"loc", "type"} for e in body["errors"])
    assert fake.calls == 0


def test_malformed_json_is_422(make_client: ClientFactory) -> None:
    client, _ = make_client()
    payload = b'{"question": "' + SECRET.encode() + b'"'
    r = client.post("/v1/qa", content=payload, headers={"content-type": "application/json"})
    assert r.status_code == 422
    assert SECRET not in r.text


def test_body_limit_413_by_content_length(make_client: ClientFactory) -> None:
    client, fake = make_client()
    r = client.post("/v1/qa", content=b"{" + b" " * 20_000 + b"}", headers={"content-type": "application/json"})
    assert r.status_code == 413
    assert r.json()["error_code"] == "payload_too_large"
    assert r.headers["x-request-id"]
    assert fake.calls == 0


def test_body_limit_413_streamed_without_content_length(make_client: ClientFactory) -> None:
    client, fake = make_client()

    def gen() -> Iterator[bytes]:
        for _ in range(10):
            yield b" " * 4096

    r = client.post("/v1/qa", content=gen(), headers={"content-type": "application/json"})
    assert r.status_code == 413
    assert fake.calls == 0


def test_body_limit_configurable(make_client: ClientFactory) -> None:
    client, _ = make_client(ServiceSettings(max_body_bytes=1024))
    assert client.post("/v1/qa", json={"question": "y" * 1900}).status_code == 413


# ---------------------------------------------------------------- error mapping
@pytest.mark.parametrize(
    ("exc", "status", "code"),
    [
        (ModeUnavailableError("finetuned", "adapter missing"), 503, "mode_unavailable"),
        (RetrieverUnavailableError("qdrant down"), 503, "artifact_unavailable"),
        (ArtifactUnavailableError("missing"), 503, "artifact_unavailable"),
        (GenerationTimeoutError("60s"), 504, "generation_timeout"),
        (GenerationError("prompt too long"), 502, "generation_failed"),
        (ContractViolationError("corpus_version mismatch"), 500, "internal_error"),
        (RuntimeError("boom /secret/path"), 500, "internal_error"),
    ],
)
def test_error_mapping(make_client: ClientFactory, exc: Exception, status: int, code: str) -> None:
    client, _ = make_client(raise_exc=exc)
    r = client.post("/v1/qa", json=Q)
    assert r.status_code == status
    body = r.json()
    assert body["error_code"] == code
    assert str(exc) not in r.text or code == "mode_unavailable"
    assert ("retry-after" in r.headers) == (status == 503)


def test_unavailable_mode_rejected_before_pipeline(make_client: ClientFactory) -> None:
    client, fake = make_client(modes=frozenset({"base", "rag"}))
    r = client.post("/v1/qa", json={**Q, "experiment_mode": "finetuned_rag"})
    assert r.status_code == 503
    assert r.json()["error_code"] == "mode_unavailable"
    assert fake.calls == 0


def test_api_backstop_timeout_504_and_slot_held_until_thread_ends(make_client: ClientFactory) -> None:
    client, fake = make_client(ServiceSettings(request_timeout_s=0.3, queue_timeout_s=0.2))
    fake.release.clear()
    r = client.post("/v1/qa", json=Q)
    assert r.status_code == 504
    assert r.json()["error_code"] == "request_timeout"
    # the timed-out generation still holds the only slot -> busy
    r2 = client.post("/v1/qa", json=Q)
    assert r2.status_code == 503
    assert r2.json()["error_code"] == "busy"
    fake.release.set()
    deadline = time.time() + 5
    while time.time() < deadline:
        r3 = client.post("/v1/qa", json=Q)
        if r3.status_code == 200:
            break
        time.sleep(0.05)
    assert r3.status_code == 200
    assert fake.max_concurrent == 1


def test_global_semaphore_serialises_generation(make_client: ClientFactory) -> None:
    client, fake = make_client(ServiceSettings(queue_timeout_s=10), delay_s=0.1)
    results: list[int] = []

    def call() -> None:
        results.append(client.post("/v1/qa", json=Q).status_code)

    threads = [threading.Thread(target=call) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [200] * 4
    assert fake.max_concurrent == 1


def test_unknown_route_uses_error_envelope(make_client: ClientFactory) -> None:
    client, _ = make_client()
    r = client.get("/nope")
    assert r.status_code == 404
    assert r.json()["error_code"] == "not_found"
    assert client.get("/v1/qa").status_code == 405
