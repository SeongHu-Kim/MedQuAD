"""Example-questions panel and behaviour check (D-069). Offline: API mocked, no network, no models.

The example questions are read from configs/service/example_questions.yaml, not repeated here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from medquad_qa.rag.safety import EMERGENCY_MESSAGE
from medquad_qa.ui import client as ui_client
from medquad_qa.ui.examples import BEHAVIOURS, Example, actual_state, compare, find_example, load_examples

APP = str(Path(__file__).resolve().parents[2] / "src" / "medquad_qa" / "ui" / "app.py")
INFO = {"available_modes": ["base", "rag"], "versions": {}, "service_version": "0", "contracts_version": "0"}
ANSWERED_CITED = {
    "request_id": "r1",
    "experiment_mode": "rag",
    "answer": "Synthetic answer.",
    "abstained": False,
    "citations": [{"record_id": "mq-0000000000000001", "evidence_snippet": "synthetic"}],
    "warnings": [],
    "model_version": "fake@0",
    "prompt_version": "p0",
    "latency_ms": 1.0,
}


def _resp(**kw: Any) -> dict[str, Any]:
    return {**ANSWERED_CITED, **kw}


def _ex(expected: dict[str, str]) -> Example:
    return Example("그룹", "Synthetic question?", expected, "기대 동작 설명")


# ---------- config ----------


def test_config_loads_all_examples_with_known_codes() -> None:
    groups = load_examples()
    assert [n for n, _ in groups] == [
        "일반 질문",
        "희귀 질환",
        "가상 질환",
        "범위 밖",
        "근거 부족",
        "개인 조언",
        "경계 (답해야 함)",
        "위기 표현",
        "알려진 한계",
    ]
    examples = [e for _, items in groups for e in items]
    assert len(examples) == 11
    for e in examples:
        assert e.expected and all(code in BEHAVIOURS for code in e.expected.values())
        assert e.expected_text


def test_config_rejects_unknown_codes_and_modes(tmp_path: Path) -> None:
    bad = tmp_path / "x.yaml"
    bad.write_text(
        'version: 1\ngroups:\n- name: g\n  examples:\n  - {question: "q?", expected: {rag: nope}, expected_text: t}\n'
    )
    with pytest.raises(ValueError, match="unknown behaviour"):
        load_examples(bad)
    bad.write_text(
        "version: 1\ngroups:\n- name: g\n  examples:\n"
        '  - {question: "q?", expected: {gpt: answered}, expected_text: t}\n'
    )
    with pytest.raises(ValueError, match="unknown mode"):
        load_examples(bad)


# ---------- comparison ----------


@pytest.mark.parametrize(
    ("code", "resp", "state", "match"),
    [
        ("answered_cited", _resp(), "answered_cited", True),
        ("answered_cited", _resp(citations=[]), "answered_uncited", False),
        ("answered_uncited", _resp(citations=[]), "answered_uncited", True),
        ("answered", _resp(citations=[]), "answered_uncited", True),
        ("answered", _resp(abstained=True, abstention_reason="insufficient_evidence"), "abstain_no_evidence", False),
        (
            "abstain_no_evidence",
            _resp(abstained=True, abstention_reason="no_relevant_evidence"),
            "abstain_no_evidence",
            True,
        ),
        (
            "abstain_no_evidence",
            _resp(abstained=True, abstention_reason="insufficient_evidence"),
            "abstain_no_evidence",
            True,
        ),
        ("abstain_no_evidence", _resp(), "answered_cited", False),
        (
            "refuse_personal",
            _resp(abstained=True, abstention_reason="personalized_medical_advice", answer="refusal"),
            "refuse_personal",
            True,
        ),
        (
            "refuse_personal",
            _resp(abstained=True, abstention_reason="personalized_medical_advice", answer=EMERGENCY_MESSAGE),
            "crisis_message",
            True,
        ),
        (
            "crisis_message",
            _resp(abstained=True, abstention_reason="personalized_medical_advice", answer=EMERGENCY_MESSAGE),
            "crisis_message",
            True,
        ),
        (
            "crisis_message",
            _resp(abstained=True, abstention_reason="personalized_medical_advice", answer="refusal"),
            "refuse_personal",
            False,
        ),
        (
            "crisis_message",
            _resp(
                abstained=True,
                abstention_reason="personalized_medical_advice",
                answer=EMERGENCY_MESSAGE,
                warnings=["safety_check_failed"],
            ),
            "safety_check_failed",
            False,
        ),
        (
            "abstain_missing_citations",
            _resp(abstained=True, abstention_reason="missing_citations"),
            "abstain_missing_citations",
            True,
        ),
        (
            "abstain_missing_citations",
            _resp(abstained=True, abstention_reason="invalid_citations"),
            "abstain:invalid_citations",
            False,
        ),
    ],
)
def test_compare_decides_match_from_observable_fields(code: str, resp: dict[str, Any], state: str, match: bool) -> None:
    assert actual_state(resp)[0] == state
    check = compare(_ex({"rag": code}), "rag", resp)
    assert check.expected_code == code
    assert check.match is match


def test_compare_all_modes_and_unspecified_mode() -> None:
    ex = _ex({"all": "answered"})
    assert compare(ex, "finetuned", _resp(citations=[])).match is True
    only_rag = _ex({"rag": "answered_cited"})
    check = compare(only_rag, "base", _resp(citations=[]))
    assert check.match is None and check.expected_code is None


def test_find_example_requires_exact_question() -> None:
    groups = load_examples()
    q = groups[0][1][0].question
    assert find_example(groups, f"  {q}  ") is not None
    assert find_example(groups, q + " please") is None


# ---------- AppTest ----------


@pytest.fixture(autouse=True)
def _fresh_client_cache() -> None:
    st.cache_resource.clear()


def _mock(monkeypatch: pytest.MonkeyPatch, qa: dict[str, Any]) -> None:
    real = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/info":
            return httpx.Response(200, json=INFO)
        if request.url.path == "/health/ready":
            return httpx.Response(200, json={"status": "ready"})
        return httpx.Response(200, json=qa)

    def fake_client(*args: object, **kwargs: object) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ui_client.httpx, "Client", fake_client)


def test_panel_lists_groups_and_click_fills_question(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock(monkeypatch, ANSWERED_CITED)
    groups = load_examples()
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    labels = [e.label for e in at.sidebar.expander]
    assert [n for n, _ in groups] == labels
    assert any("예시 질문" in h.value for h in at.sidebar.header)
    first = groups[0][1][0].question
    at.button(key="example_0_0").click().run()
    assert at.text_area(key="question").value == first


def test_check_box_shows_match_for_example_and_not_for_free_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock(monkeypatch, ANSWERED_CITED)  # rag answer with one citation
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.button(key="example_0_0").click().run()  # 일반 질문: rag expects a cited answer (default mode is rag)
    at.button(key="ask").click().run()
    assert not at.exception
    assert any("동작 확인 · 정확도 판정 아님" in s.value for s in at.subheader)
    assert any("기대한 동작과 일치" in s.value for s in at.success)  # Streamlit renders the ✅ as the icon

    at.text_area(key="question").input("A free-text synthetic question?").run()
    at.button(key="ask").click().run()
    assert not any("동작 확인" in s.value for s in at.subheader)


def test_check_box_shows_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock(monkeypatch, _resp(abstained=True, abstention_reason="insufficient_evidence", citations=[]))
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.button(key="example_0_0").click().run()
    at.button(key="ask").click().run()
    assert any("기대한 동작과 다름" in w.value for w in at.warning)  # ⚠️ is rendered as the icon
