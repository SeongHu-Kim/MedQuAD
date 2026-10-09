"""Streamlit demo UI. Run: ``streamlit run src/medquad_qa/ui/app.py`` (API at MEDQUAD_API_URL).

Owner: service-platform-engineer. All model/evidence text is rendered escaped (no Markdown/HTML).
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from medquad_qa.ui.client import ApiClient, answer_view, md_escape, mode_options
from medquad_qa.ui.examples import Example, compare, find_example, load_examples

MAX_QUESTION_CHARS = 2000
QUESTION_KEY = "question"
CHECK_TITLE = "동작 확인 · 정확도 판정 아님"
DISCLAIMER = (
    "연구용 프로토타입으로, 일반적인 의료 정보 제공만을 목적으로 합니다. 의료 조언이 아니며 임상적으로 검증되지 "
    "않았습니다. 자격을 갖춘 의료 전문가와 상담하세요. 개인 건강 정보를 입력하지 마세요."
)
ASK_LABEL = "질문하기 (Ask)"


@st.cache_resource
def _client() -> ApiClient:
    return ApiClient()


def _fill_question(text: str) -> None:
    st.session_state[QUESTION_KEY] = text


def _examples_panel(groups: list[tuple[str, list[Example]]]) -> None:
    st.header("예시 질문")
    st.caption(f"질문을 누르면 입력란에 채워집니다. 모드를 고른 뒤 '{ASK_LABEL}'를 누르세요.")
    for g_idx, (name, items) in enumerate(groups):
        with st.expander(name):
            for e_idx, ex in enumerate(items):
                st.button(
                    ex.question,
                    key=f"example_{g_idx}_{e_idx}",
                    on_click=_fill_question,
                    args=(ex.question,),
                    use_container_width=True,
                )
                st.caption(f"기대 동작: {ex.expected_text}")


def _behaviour_check(example: Example, mode: str, resp: dict[str, Any]) -> None:
    check = compare(example, mode, resp)
    with st.container(border=True):
        st.subheader(CHECK_TITLE)
        st.text(f"기대한 동작: {example.expected_text}")
        st.text(f"이 모드({mode})의 기대 동작: {check.expected_label}")
        st.text(f"실제 동작: {check.actual_label}")
        for d in check.details:
            st.text(d)
        if check.match is None:
            st.info("비교하지 않음: 이 모드에 대해 정해진 기대 동작이 없습니다.")
        elif check.match:
            st.success("✅ 기대한 동작과 일치")
        else:
            st.warning("⚠️ 기대한 동작과 다름")
        if example.note:
            st.caption(f"참고: {example.note}")
        st.caption(
            "관찰 가능한 동작(답변/거부, 거부 사유, 위기 안내, 인용 유무)만 비교하며, "
            "답변의 정확도는 판정하지 않습니다."
        )


def main() -> None:
    st.set_page_config(page_title="MedQuAD 근거 기반 의료 QA", layout="wide")
    st.title("MedQuAD 근거 기반 의료 QA")
    st.warning(DISCLAIMER)

    client = _client()
    info = client.info()
    ready = client.ready()

    # Sidebar, top to bottom: mode, top_k, examples, service status, loaded versions (D-070).
    with st.sidebar:
        avail = info.data.get("available_modes", []) if info.ok else []
        options = mode_options(avail)
        labels = [f"{label}{'' if ok else '  [사용 불가]'}" for _, label, ok in options]
        default = next((i for i, (m, _, ok) in enumerate(options) if m == "rag" and ok), 0)
        choice = st.radio("모드 선택", labels, index=default)
        mode = options[labels.index(choice)][0]  # the API value (base / rag / finetuned / finetuned_rag)
        top_k = st.slider("근거 레코드 수 (top_k)", 1, 20, 5, disabled=mode not in ("rag", "finetuned_rag"))
        groups = load_examples()
        _examples_panel(groups)
        st.header("서비스 상태")
        if ready.ok:
            st.success("API 준비 완료 (ready)")
        else:
            st.error(ready.error or "API가 준비되지 않았습니다")
        if info.ok:
            st.subheader("로드된 버전")
            for k, v in (info.data.get("versions") or {}).items():
                st.text(f"{k}: {v or '-'}")
            st.caption(f"API {info.data.get('service_version')} / contracts {info.data.get('contracts_version')}")

    question = st.text_area(
        "질문",
        max_chars=MAX_QUESTION_CHARS,
        height=100,
        placeholder="예: What are the symptoms of glaucoma?",
        key=QUESTION_KEY,
    )
    if st.button(ASK_LABEL, type="primary", disabled=not question.strip(), key="ask"):
        with st.spinner("답변 생성 중..."):
            result = client.ask(question.strip(), mode, top_k)
        if not result.ok:
            st.error(result.error or "요청에 실패했습니다")
            if result.data.get("request_id"):
                st.caption(f"request_id: {result.data['request_id']}")
            return
        view = answer_view(result.data)
        # Main panel after asking: mode label, notices + answer (+ citations), mode description,
        # behaviour check (unedited examples only), run metadata (D-070).
        st.subheader(view["mode_label"])
        for level, text in view["notices"]:
            (st.warning if level == "warning" else st.info)(text)
        st.markdown(md_escape(view["answer"]))
        if view["citations"]:
            st.subheader("인용 (검색된 레코드)")
            for c in view["citations"]:
                title = " / ".join(x for x in (c["record_id"], c["topic"], c["source"]) if x)
                with st.expander(title):
                    st.markdown(md_escape(c["snippet"]))
                    if c["url"]:
                        st.text(f"출처 URL: {c['url']}")
        if view["mode_note"]:
            st.info(view["mode_note"])
        example = find_example(groups, question)
        if example is not None:
            _behaviour_check(example, mode, result.data)
        st.subheader("실행 정보")
        meta = {**view["versions"], "request_id": view["request_id"], "latency_ms": view["latency_ms"]}
        st.json(meta)
        if view["component_latency_ms"]:
            st.json({"component_latency_ms": view["component_latency_ms"]})
        st.caption(DISCLAIMER)


main()
