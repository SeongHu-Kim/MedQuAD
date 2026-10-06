"""Answerability-predictor contracts (non-clinical: does the evidence suffice to answer?).

Owner: lead. Kept separate so qa.py and evaluation.py can both import it without a cycle.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

NegativeType = Literal["easy_random", "same_topic_diff_qtype", "lexical_hard_bm25"]


class PredictorOutput(BaseModel):
    """Answerability of a (question, evidence list) pair. NOT diagnostic confidence.

    Aggregation over evidence is max(per_evidence_scores); an empty evidence list gives 0.0 / False.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    answerability_score: float = Field(ge=0.0, le=1.0)
    predicted_label: bool
    threshold_version: str
    model_version: str
    per_evidence_scores: list[float] = Field(default_factory=list)


class AnswerabilityPair(BaseModel):
    """One synthetic pair for the answerability classifier.

    Pairs are built within a single split (evidence pool from the same split).
    Train pairs train; validation pairs select/calibrate/threshold; test pairs are for
    reporting classifier metrics only. Never built from frozen evaluation items.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pair_id: str
    question: str
    question_record_id: str
    evidence_record_id: str
    evidence_text: str
    label: bool
    negative_type: NegativeType | None = Field(default=None, description="None for positive pairs.")
    label_provenance: Literal["synthetic_rule"] = "synthetic_rule"
    pair_rule_version: str
    split: Literal["train", "validation", "test"]
    split_group_id: str
