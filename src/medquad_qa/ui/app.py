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
    "Research prototype for general medical information only. Not medical advice; not clinically validated. "
    "Consult a qualified clinician. Do not enter personal health information."
)


@st.cache_resource
def _client() -> ApiClient:
    return ApiClient()


def _fill_question(text: str) -> None:
    st.session_state[QUESTION_KEY] = text


def _examples_panel(groups: list[tuple[str, list[Example]]]) -> None:
    st.header("예시 질문")
    st.caption("질문을 누르면 입력란에 채워집니다. 모드를 고른 뒤 'Ask'를 누르세요.")
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
    st.set_page_config(page_title="MedQuAD Evidence-Grounded QA", layout="wide")
    st.title("MedQuAD Evidence-Grounded QA")
    st.warning(DISCLAIMER)

    client = _client()
    info = client.info()
    ready = client.ready()

    with st.sidebar:
        st.header("Service")
        if ready.ok:
            st.success("API ready")
        else:
            st.error(ready.error or "API not ready")
        avail = info.data.get("available_modes", []) if info.ok else []
        options = mode_options(avail)
        labels = [f"{label}{'' if ok else '  [unavailable]'}" for _, label, ok in options]
        default = next((i for i, (m, _, ok) in enumerate(options) if m == "rag" and ok), 0)
        choice = st.radio("Experiment mode", labels, index=default)
        mode = options[labels.index(choice)][0]
        top_k = st.slider("Evidence records (top_k)", 1, 20, 5, disabled=mode not in ("rag", "finetuned_rag"))
        if info.ok:
            st.subheader("Loaded versions")
            for k, v in (info.data.get("versions") or {}).items():
                st.text(f"{k}: {v or '-'}")
            st.caption(f"API {info.data.get('service_version')} / contracts {info.data.get('contracts_version')}")
        groups = load_examples()
        _examples_panel(groups)

    question = st.text_area(
        "Question",
        max_chars=MAX_QUESTION_CHARS,
        height=100,
        placeholder="e.g. What are the symptoms of glaucoma?",
        key=QUESTION_KEY,
    )
    if st.button("Ask", type="primary", disabled=not question.strip(), key="ask"):
        with st.spinner("Generating..."):
            result = client.ask(question.strip(), mode, top_k)
        if not result.ok:
            st.error(result.error or "Request failed")
            if result.data.get("request_id"):
                st.caption(f"request_id: {result.data['request_id']}")
            return
        view = answer_view(result.data)
        example = find_example(groups, question)
        if example is not None:
            _behaviour_check(example, mode, result.data)
        st.subheader(view["mode_label"])
        for level, text in view["notices"]:
            (st.warning if level == "warning" else st.info)(text)
        st.markdown(md_escape(view["answer"]))
        if view["citations"]:
            st.subheader("Citations (retrieved records)")
            for c in view["citations"]:
                title = " / ".join(x for x in (c["record_id"], c["topic"], c["source"]) if x)
                with st.expander(title):
                    st.markdown(md_escape(c["snippet"]))
                    if c["url"]:
                        st.text(f"Source URL: {c['url']}")
        st.subheader("Run metadata")
        meta = {**view["versions"], "request_id": view["request_id"], "latency_ms": view["latency_ms"]}
        st.json(meta)
        if view["component_latency_ms"]:
            st.json({"component_latency_ms": view["component_latency_ms"]})
        st.caption(view["disclaimer"])


main()
