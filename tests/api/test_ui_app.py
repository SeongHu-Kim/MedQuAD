"""Offline Streamlit AppTest of the demo page (API mocked; no network, no models)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from medquad_qa.ui import client as ui_client

APP = str(Path(__file__).resolve().parents[2] / "src" / "medquad_qa" / "ui" / "app.py")
INFO = {
    "available_modes": ["base", "rag"],
    "versions": {"model_version": "fake@0"},
    "service_version": "0.1.0",
    "contracts_version": "1.1.1",
}
QA = {
    "request_id": "r1",
    "experiment_mode": "rag",
    "answer": "Synthetic answer ![img](http://evil.example/x.png)",
    "abstained": False,
    "citations": [{"record_id": "mq-0000000000000001", "evidence_snippet": "synthetic"}],
    "retrieved_record_ids": ["mq-0000000000000001"],
    "invalid_citation_ids": ["E9"],
    "warnings": [],
    "model_version": "fake@0",
    "prompt_version": "p0",
    "latency_ms": 5.0,
    "disclaimer": "Not medical advice",
}


def _handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1/info":
        return httpx.Response(200, json=INFO)
    if request.url.path == "/health/ready":
        return httpx.Response(200, json={"status": "ready"})
    return httpx.Response(200, json=QA)


@pytest.fixture(autouse=True)
def _fresh_client_cache() -> None:
    st.cache_resource.clear()  # the page caches its ApiClient process-wide


@pytest.fixture
def mocked_api(monkeypatch: pytest.MonkeyPatch) -> None:
    real = httpx.Client

    def fake_client(*args: object, **kwargs: object) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(_handler)
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ui_client.httpx, "Client", fake_client)


def test_page_renders_disclaimer_and_modes(mocked_api: None) -> None:
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("Not medical advice" in w.value for w in at.warning)
    labels = at.radio[0].options
    assert any("finetuned" in o and "[unavailable]" in o for o in labels)


def test_ask_shows_answer_citations_and_invalid_warning(mocked_api: None) -> None:
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_area[0].input("What is synthetic X?").run()
    at.button[0].click().run()
    assert not at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "Synthetic answer" in md and "![img](" not in md
    assert any("did not match" in w.value for w in at.warning)
    assert any("mq-0000000000000001" in e.label for e in at.expander)


def test_unreachable_api_shows_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEDQUAD_API_URL", "http://127.0.0.1:9")
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("unreachable" in e.value for e in at.error)
