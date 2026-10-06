"""Evaluation contracts.

Owner: lead.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from medquad_qa.contracts.answerability import PredictorOutput
from medquad_qa.contracts.qa import AbstentionReason

EvaluationTrack = Literal["corpus_grounded", "finetune_generalization", "answerability_robustness"]

LabelProvenance = Literal[
    "human_reviewed",  # a person read and confirmed the label (non-clinical); record reviewer role, not identity
    "synthetic_rule",  # produced by a documented deterministic rule
    "llm_generated_unreviewed",  # written by an AI agent/model, not checked by a person
    "llm_generated_reviewed",  # written by an AI agent/model and checked by a person
    "derived_from_dataset",  # e.g. the original MedQuAD answer as reference
]

CaseType = Literal[
    "answerable", "unanswerable", "ambiguous", "conflicting_evidence", "personalized_advice", "adversarial"
]


class EvaluationExample(BaseModel):
    """One frozen evaluation item.

    ``gold_record_ids`` is a set: retrieving ANY of them counts as relevant.
    Committed evaluation files carry IDs, questions and labels only; reference answers
    are resolved from the corpus at runtime (no dataset text in committed eval files).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    example_id: str
    question: str
    reference_answer: str | None = None
    gold_record_ids: list[str] = Field(default_factory=list)
    graded_relevance: dict[str, int] | None = Field(default=None, description="nDCG only when present.")
    answerable: bool
    expected_behavior: Literal["answer", "abstain", "either"] = "answer"
    expected_abstention_reason: AbstentionReason | None = None
    label_provenance: LabelProvenance
    question_provenance: LabelProvenance
    paraphrase_method: str | None = Field(default=None, description="e.g. 'template_rule_v1', 'agent_written'.")
    reviewer: Literal["none", "ai_agent", "human_nonclinical"] = "none"
    derived_from_record_ids: list[str] = Field(default_factory=list)
    split_group_id: str | None = Field(default=None, description="Group of the source record, for leakage checks.")
    eval_split: Literal["dev", "test", "train_probe"] = Field(
        description=(
            "train_probe items come from TRAIN groups by design (Track B memorization probe). They are never pooled "
            "with test metrics and are exempt only from the test-group-origin rule, not from other leakage checks."
        )
    )
    evaluation_track: EvaluationTrack
    case_type: CaseType = "answerable"
    question_type: str | None = None
    topic: str | None = None
    injected_evidence: list[str] | None = Field(
        default=None, description="Synthetic evidence for conflict/injection fixtures (served by a fixture retriever)."
    )
    notes: str | None = None


__all__ = [
    "CaseType",
    "EvaluationExample",
    "EvaluationTrack",
    "LabelProvenance",
    "PredictorOutput",
]
