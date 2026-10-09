"""Korean GUI layout and API values (D-070). Offline: API mocked, no network, no models."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from medquad_qa.ui import client as ui_client
from medquad_qa.ui.client import MODE_LABELS

APP = str(Path(__file__).resolve().parents[2] / "src" / "medquad_qa" / "ui" / "app.py")
INFO = {
    "available_modes": ["base", "rag", "finetuned", "finetuned_rag"],
    "versions": {"model_version": "fake@0"},
    "service_version": "0",
    "contracts_version": "0",
}
QA = {
    "request_id": "r1",
    "experiment_mode": "base",
    "answer": "Synthetic answer.",
    "abstained": False,
    "citations": [],
    "warnings": [],
    "model_version": "fake@0",
    "prompt_version": "p0",
    "latency_ms": 1.0,
}


@pytest.fixture(autouse=True)
def _fresh_client_cache() -> None:
    st.cache_resource.clear()


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    real = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/info":
            return httpx.Response(200, json=INFO)
        if request.url.path == "/health/ready":
            return httpx.Response(200, json={"status": "ready"})
        body = json.loads(request.content)
        payloads.append(body)
        return httpx.Response(200, json={**QA, "experiment_mode": body["experiment_mode"]})

    def fake_client(*args: object, **kwargs: object) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ui_client.httpx, "Client", fake_client)
    return payloads


def _walk(node: Any) -> Iterator[Any]:
    for child in getattr(node, "children", {}).values():
        yield child
        yield from _walk(child)


def _texts(node: Any) -> list[str]:
    out = []
    for e in _walk(node):
        v = (
            getattr(e, "label", None)
            if type(e).__name__ in ("Radio", "Slider", "Expander", "Button")
            else getattr(e, "value", "")
        )
        out.append(f"{type(e).__name__}:{v}")
    return out


def _index(items: list[str], prefix: str) -> int:
    return next(i for i, s in enumerate(items) if s.startswith(prefix))


def test_sidebar_order(sent: list[dict[str, Any]]) -> None:
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    items = _texts(at.sidebar)
    order = [
        _index(items, "Header:모드 선택"),
        _index(items, "Button:**기본 모델 (base):**"),
        _index(items, "Button:**미세조정 + 검색 (finetuned_rag):**"),
        _index(items, "Slider:근거 레코드 수 (top_k)"),
        _index(items, "Header:예시 질문"),
        _index(items, "Header:서비스 상태"),
        _index(items, "Subheader:로드된 버전"),
    ]
    assert order == sorted(order)
    assert any(s.startswith("Success:API 준비 완료") for s in items)


def test_main_panel_order_after_asking_an_example(sent: list[dict[str, Any]]) -> None:
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.button(key="mode_base").click().run()
    at.button(key="example_0_0").click().run()  # 일반 질문: base expects an uncited answer
    at.button(key="ask").click().run()
    assert not at.exception
    items = _texts(at.main)
    order = [
        _index(items, "Title:"),
        _index(items, "Warning:연구용 프로토타입"),
        _index(items, "TextArea:"),
        _index(items, "Button:"),
        _index(items, f"Subheader:{MODE_LABELS['base']}"),
        _index(items, "Markdown:Synthetic answer"),
        _index(items, "Info:closed-book 모드"),
        _index(items, "Subheader:동작 확인 · 정확도 판정 아님"),
        _index(items, "Subheader:실행 정보"),
    ]
    assert order == sorted(order)


@pytest.mark.parametrize("mode", ["base", "rag", "finetuned", "finetuned_rag"])
def test_api_receives_unchanged_mode_values(sent: list[dict[str, Any]], mode: str) -> None:
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.button(key=f"mode_{mode}").click().run()
    at.text_area(key="question").input("A synthetic question?").run()
    at.button(key="ask").click().run()
    assert not at.exception
    assert sent[-1]["experiment_mode"] == mode
    assert set(sent[-1]) == {"question", "experiment_mode", "top_k"}


def test_mode_buttons_default_selection_and_labels(sent: list[dict[str, Any]]) -> None:
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.session_state["mode"] == "rag"  # default when available
    types = {m: at.button(key=f"mode_{m}").proto.type for m in MODE_LABELS}
    assert types == {"base": "secondary", "rag": "primary", "finetuned": "secondary", "finetuned_rag": "secondary"}
    assert at.button(key="mode_base").label == "**기본 모델 (base):**  \n검색 없음 (closed-book)"
    at.button(key="mode_finetuned").click().run()
    assert at.session_state["mode"] == "finetuned"
    assert at.button(key="mode_finetuned").proto.type == "primary"
    assert at.button(key="mode_rag").proto.type == "secondary"


def test_unavailable_modes_disabled_and_default_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    real = httpx.Client
    info = {**INFO, "available_modes": ["base", "finetuned"]}  # rag unavailable

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/info":
            return httpx.Response(200, json=info)
        return httpx.Response(200, json={"status": "ready"})

    def fake_client(*args: object, **kwargs: object) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ui_client.httpx, "Client", fake_client)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.session_state["mode"] == "base"  # first available mode when rag is unavailable
    for m in ("rag", "finetuned_rag"):
        b = at.button(key=f"mode_{m}")
        assert b.disabled and "[사용 불가]" in b.label
    assert not at.button(key="mode_finetuned").disabled
