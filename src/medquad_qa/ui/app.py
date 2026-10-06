"""Streamlit demo UI. Run: ``streamlit run src/medquad_qa/ui/app.py`` (API at MEDQUAD_API_URL).

Owner: service-platform-engineer. All model/evidence text is rendered escaped (no Markdown/HTML).
"""

from __future__ import annotations

import streamlit as st

from medquad_qa.ui.client import ApiClient, answer_view, md_escape, mode_options

MAX_QUESTION_CHARS = 2000
DISCLAIMER = (
    "Research prototype for general medical information only. Not medical advice; not clinically validated. "
    "Consult a qualified clinician. Do not enter personal health information."
)


@st.cache_resource
def _client() -> ApiClient:
    return ApiClient()


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

    question = st.text_area(
        "Question", max_chars=MAX_QUESTION_CHARS, height=100, placeholder="e.g. What are the symptoms of glaucoma?"
    )
    if st.button("Ask", type="primary", disabled=not question.strip()):
        with st.spinner("Generating..."):
            result = client.ask(question.strip(), mode, top_k)
        if not result.ok:
            st.error(result.error or "Request failed")
            if result.data.get("request_id"):
                st.caption(f"request_id: {result.data['request_id']}")
            return
        view = answer_view(result.data)
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
