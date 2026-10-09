"""HTTP client and view-model helpers for the demo UI (no Streamlit import, unit-testable).

Owner: service-platform-engineer. The UI talks to the API over HTTP only; it never loads models.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import httpx

# Display text only (D-070): the keys are the experiment_mode values sent to the API and never change.
MODE_LABELS: dict[str, str] = {
    "base": "기본 모델 (base): 검색 없음 (closed-book)",
    "rag": "검색 기반 모델 (rag): 기본 모델 + 검색된 근거 (인용 포함)",
    "finetuned": "미세조정 모델 (finetuned): LoRA 미세조정, 검색 없음 (closed-book)",
    "finetuned_rag": "미세조정 + 검색 (finetuned_rag): LoRA 미세조정 모델 + 검색된 근거 (인용 포함)",
}
CLOSED_BOOK_NOTE = "closed-book 모드: 근거를 검색하지 않으므로 답변에 인용이 없습니다."
RAG_MODES = frozenset({"rag", "finetuned_rag"})
DEFAULT_TIMEOUT_S = 130.0  # a little above the API's 120 s backstop


def api_url() -> str:
    return os.environ.get("MEDQUAD_API_URL", "http://127.0.0.1:8000").rstrip("/")


@dataclass
class ApiResult:
    ok: bool
    status: int
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class ApiClient:
    def __init__(self, base_url: str | None = None, transport: httpx.BaseTransport | None = None) -> None:
        self._client = httpx.Client(base_url=base_url or api_url(), timeout=DEFAULT_TIMEOUT_S, transport=transport)

    def _get(self, path: str) -> ApiResult:
        try:
            r = self._client.get(path, timeout=5.0)
        except httpx.HTTPError as exc:
            return ApiResult(False, 0, error=f"API에 연결할 수 없습니다 ({type(exc).__name__})")
        return _result(r)

    def info(self) -> ApiResult:
        return self._get("/v1/info")

    def ready(self) -> ApiResult:
        return self._get("/health/ready")

    def ask(self, question: str, mode: str, top_k: int) -> ApiResult:
        payload = {"question": question, "experiment_mode": mode, "top_k": top_k}
        try:
            r = self._client.post("/v1/qa", json=payload)
        except httpx.TimeoutException:
            return ApiResult(False, 0, error="요청 시간이 초과되었습니다.")
        except httpx.HTTPError as exc:
            return ApiResult(False, 0, error=f"API에 연결할 수 없습니다 ({type(exc).__name__})")
        return _result(r)


def _result(r: httpx.Response) -> ApiResult:
    try:
        data = r.json()
    except ValueError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    if r.is_success:
        return ApiResult(True, r.status_code, data)
    code = data.get("error_code", "http_error")
    msg = data.get("message", r.reason_phrase)
    return ApiResult(False, r.status_code, data, error=f"{r.status_code} {code}: {msg}")


def mode_options(available: list[str] | None) -> list[tuple[str, str, bool]]:
    """(mode, label, available) for every experiment mode, in a fixed order."""
    avail = set(available or [])
    return [(m, label, m in avail) for m, label in MODE_LABELS.items()]


def answer_view(resp: dict[str, Any]) -> dict[str, Any]:
    """Flatten a QAResponse JSON into what the page shows. Pure function (tested offline)."""
    mode = resp.get("experiment_mode", "")
    citations = [
        {
            "record_id": c.get("record_id"),
            "topic": c.get("topic") or "",
            "source": c.get("source_name") or "",
            "url": c.get("source_url") or "",
            "snippet": c.get("evidence_snippet", ""),
        }
        for c in resp.get("citations", [])
    ]
    notices: list[tuple[str, str]] = []
    if resp.get("abstained"):
        notices.append(
            ("info", f"시스템이 답변을 거부했습니다 (abstained): {resp.get('abstention_reason') or '미지정'}")
        )
    if resp.get("invalid_citation_ids"):
        notices.append(
            (
                "warning",
                f"모델이 생성한 인용 {len(resp['invalid_citation_ids'])}건이 제공된 근거와 일치하지 않아 "
                "제거되었습니다.",
            )
        )
    if mode in RAG_MODES and not resp.get("abstained") and not citations:
        notices.append(("warning", "이 RAG 답변에는 유효한 인용이 없습니다."))
    for w in resp.get("warnings", []):
        notices.append(("warning", f"파이프라인 경고: {w}"))
    versions = {
        k: resp.get(k)
        for k in ("model_version", "corpus_version", "index_version", "prompt_version", "retriever")
        if resp.get(k)
    }
    return {
        "mode": mode,
        "mode_label": MODE_LABELS.get(mode, mode),
        "answer": resp.get("answer", ""),
        "abstained": bool(resp.get("abstained")),
        "citations": citations,
        "notices": notices,
        "mode_note": CLOSED_BOOK_NOTE if mode not in RAG_MODES else "",
        "versions": versions,
        "latency_ms": resp.get("latency_ms"),
        "component_latency_ms": resp.get("component_latency_ms", {}),
        "request_id": resp.get("request_id"),
        "disclaimer": resp.get("disclaimer", ""),
    }


_MD_SPECIAL = set("\\`*_{}[]()#+-.!|<>~=$")


def md_escape(text: str) -> str:
    """Escape Markdown so model/evidence text renders literally (no links, images or formatting)."""
    return "".join("\\" + ch if ch in _MD_SPECIAL else ch for ch in text)
