"""Offline tests for lead-owned shared contracts (synthetic values only)."""

import pytest
from pydantic import ValidationError

from medquad_qa.contracts import (
    CONTRACTS_VERSION,
    Citation,
    ComponentStatus,
    MedicalRecord,
    ModeUnavailableError,
    PredictorOutput,
    Provenance,
    QARequest,
    QAResponse,
    RetrievalHit,
)

SHA = "0" * 64


def test_version() -> None:
    assert CONTRACTS_VERSION == "1.1.1"


@pytest.mark.parametrize("q", ["   ", "a\x00bcd", "ab", "x" * 2001, "bad\x1bescape"])
def test_question_rejected(q: str) -> None:
    with pytest.raises(ValidationError):
        QARequest(question=q)


def test_question_stripped_and_newlines_allowed() -> None:
    r = QARequest(question="  What is glaucoma?\n  ")
    assert r.question == "What is glaucoma?"
    assert QARequest(question="line one\nline two\tok").top_k == 5


@pytest.mark.parametrize("field,value", [("top_k", 0), ("top_k", 21), ("experiment_mode", "clinical")])
def test_request_bounds(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        QARequest(question="What is glaucoma?", **{field: value})


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        QARequest(question="What is glaucoma?", debug=True)  # type: ignore[call-arg]


def test_record_id_pattern() -> None:
    prov = Provenance(
        dataset_name="synthetic", source_file="x.csv", source_file_sha256=SHA, row_index=0, pipeline_version="t"
    )
    kw = dict(
        question="q",
        answer="a",
        question_raw="q",
        answer_raw="a",
        provenance=prov,
        content_hash="h",
        duplicate_group_id="dg",
        split_group_id="sg",
    )
    MedicalRecord(record_id="mq-0123456789abcdef", **kw)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        MedicalRecord(record_id="mq-123", **kw)  # type: ignore[arg-type]


def test_response_roundtrip() -> None:
    hit = RetrievalHit(
        record_id="mq-0123456789abcdef",
        rank=1,
        score=3.2,
        retriever="bm25:answer",
        evidence_text="synthetic",
        corpus_version="c1",
    )
    resp = QAResponse(
        request_id="r1",
        experiment_mode="rag",
        answer="Synthetic [mq-0123456789abcdef].",
        abstained=False,
        citations=[Citation(record_id=hit.record_id, evidence_snippet="synthetic")],
        retrieved_record_ids=[hit.record_id],
        answerability=PredictorOutput(
            answerability_score=0.9, predicted_label=True, threshold_version="t", model_version="m"
        ),
        model_version="m@rev",
        prompt_version="rag-v1",
        latency_ms=1.0,
    )
    assert QAResponse.model_validate_json(resp.model_dump_json()) == resp


def test_mode_unavailable_is_artifact_error() -> None:
    from medquad_qa.contracts import ArtifactUnavailableError

    err = ModeUnavailableError("finetuned", "adapter missing")
    assert isinstance(err, ArtifactUnavailableError) and err.mode == "finetuned"
    assert ComponentStatus(name="generator:finetuned", ok=False, required=False).degraded is False
