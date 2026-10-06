"""Answer correctness/completeness rubric (0-2 each), grader provenance and agreement.

The rubric compares an answer with the reference record(s) for the same topic and question type. It measures
agreement with MedQuAD reference text, not clinical correctness. Graders:
- ``llm_judge:<model_id>@<revision>``: provisional; self-preference risk when the judge shares a family with the
  generator must be disclosed.
- ``ai_agent``: a Claude agent check; never labelled human.
- ``human_nonclinical``: only when the user actually fills in the spot-check sheet. Never "clinician".
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.metrics import cohen_kappa_score

RUBRIC_VERSION = "rubric-v1"

RUBRIC_TEXT = """\
Correctness (compared with the reference answer only; not clinical judgement)
  2 = every medical statement is consistent with the reference; no contradiction or fabricated specifics
  1 = mostly consistent; at most one minor inaccuracy or one unsupported specific detail
  0 = contradicts the reference, contains a material error, or answers a different question
Completeness (relative to the reference's key points for THIS question type)
  2 = covers the reference's key points for the question
  1 = covers some key points; notable omissions
  0 = covers none of the key points, or is empty/off-topic
Abstentions are not graded on this rubric; they are scored by the abstention metrics.
"""


class RubricScore(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    example_id: str
    experiment_mode: str
    correctness: int = Field(ge=0, le=2)
    completeness: int = Field(ge=0, le=2)
    grader: str
    rubric_version: str = RUBRIC_VERSION
    provisional: bool = True
    rationale: str | None = Field(default=None, max_length=500)

    @field_validator("grader")
    @classmethod
    def _grader(cls, v: str) -> str:
        if v in ("ai_agent", "human_nonclinical") or v.startswith("llm_judge:"):
            return v
        raise ValueError("grader must be 'ai_agent', 'human_nonclinical' or 'llm_judge:<model>@<rev>'")


def aggregate(scores: Sequence[RubricScore]) -> dict[str, float | int]:
    if not scores:
        raise ValueError("no rubric scores")
    n = len(scores)
    return {
        "n": n,
        "correctness_mean": sum(s.correctness for s in scores) / n,
        "completeness_mean": sum(s.completeness for s in scores) / n,
        "correct_2_rate": sum(s.correctness == 2 for s in scores) / n,
        "complete_2_rate": sum(s.completeness == 2 for s in scores) / n,
    }


def agreement(a: Sequence[int], b: Sequence[int]) -> dict[str, float | int | None]:
    """Exact agreement and quadratic-weighted Cohen's kappa between two graders on the same items."""
    if len(a) != len(b) or not a:
        raise ValueError("paired grades must be equal-length and non-empty")
    exact = sum(x == y for x, y in zip(a, b, strict=True)) / len(a)
    kappa = None if len(set(a) | set(b)) < 2 else float(cohen_kappa_score(a, b, labels=[0, 1, 2], weights="quadratic"))
    return {"n": len(a), "exact_agreement": exact, "weighted_kappa": kappa}
