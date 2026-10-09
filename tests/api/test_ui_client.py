"""Offline tests for the UI's HTTP client and view model (synthetic data, mocked transport)."""

from __future__ import annotations

import httpx

from medquad_qa.ui.client import ApiClient, answer_view, md_escape, mode_options

RESP = {
    "request_id": "r1",
    "experiment_mode": "rag",
    "answer": "Synthetic answer [mq-0000000000000001].",
    "abstained": False,
    "citations": [{"record_id": "mq-0000000000000001", "topic": "Synthetic", "evidence_snippet": "e"}],
    "retrieved_record_ids": ["mq-0000000000000001"],
    "invalid_citation_ids": ["E7"],
    "warnings": ["lexical_fallback"],
    "model_version": "fake@0",
    "prompt_version": "rag-v1+0",
    "index_version": "idx0",
    "latency_ms": 10.0,
    "disclaimer": "Not medical advice",
}


def test_answer_view_shows_evidence_filtered_warning() -> None:
    v = answer_view({**RESP, "warnings": ["evidence_filtered"], "invalid_citation_ids": []})
    assert ("warning", "파이프라인 경고: evidence_filtered") in v["notices"]


def test_answer_view_flags_invalid_citations_and_versions() -> None:
    v = answer_view(RESP)
    assert v["citations"][0]["record_id"] == "mq-0000000000000001"
    assert any("일치하지 않아" in t for _, t in v["notices"])
    assert any("lexical_fallback" in t for _, t in v["notices"])
    assert v["versions"]["index_version"] == "idx0"
    assert v["mode_label"].startswith("검색 기반 모델 (rag)")
    assert v["mode_note"] == ""


def test_answer_view_closed_book_and_abstention() -> None:
    v = answer_view({**RESP, "experiment_mode": "base", "citations": [], "invalid_citation_ids": [], "warnings": []})
    assert "closed-book" in v["mode_note"]
    assert not any("closed-book" in t for _, t in v["notices"])
    v = answer_view({**RESP, "abstained": True, "abstention_reason": "insufficient_evidence", "citations": []})
    assert any("insufficient_evidence" in t for _, t in v["notices"])


def test_rag_answer_without_citation_warns() -> None:
    v = answer_view({**RESP, "citations": [], "invalid_citation_ids": []})
    assert any("유효한 인용이 없습니다" in t for _, t in v["notices"])


def test_mode_options_marks_unavailable() -> None:
    opts = mode_options(["base", "rag"])
    assert [m for m, _, _ in opts] == ["base", "rag", "finetuned", "finetuned_rag"]
    assert [label for _, label, _ in opts][0] == "기본 모델 (base): 검색 없음 (closed-book)"
    assert all(f"({m})" in label for m, label, _ in opts)  # Korean label + unchanged identifier
    assert [ok for _, _, ok in opts] == [True, True, False, False]


def test_md_escape_neutralises_links_and_images() -> None:
    out = md_escape("![x](http://evil.example/p.png) [click](http://evil.example) <b>")
    assert "![" not in out and "](" not in out and "<b>" not in out


def test_client_maps_errors_and_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/qa":
            return httpx.Response(503, json={"error_code": "mode_unavailable", "message": "m", "request_id": "r"})
        return httpx.Response(200, json={"available_modes": ["rag"]})

    c = ApiClient("http://api.test", transport=httpx.MockTransport(handler))
    r = c.ask("What is synthetic X?", "finetuned", 5)
    assert not r.ok and r.status == 503 and "mode_unavailable" in (r.error or "")
    assert c.info().data["available_modes"] == ["rag"]

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    c2 = ApiClient("http://api.test", transport=httpx.MockTransport(down))
    r2 = c2.ready()
    assert not r2.ok and r2.status == 0 and "연결할 수 없습니다" in (r2.error or "")
