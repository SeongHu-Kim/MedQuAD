"""Answerability gate: trained predictor if available, else a lexical-coverage heuristic.

The heuristic threshold must be fitted on VALIDATION data only (D-018) with ``fit_threshold`` and stored
in ``configs/retrieval/gate_heuristic.json``. Without a fitted file the heuristic only rejects evidence
with zero content-word overlap and reports ``threshold_version='heuristic-unfitted'``.
Scores here are answerability signals, never diagnostic confidence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from medquad_qa.contracts import AnswerabilityPredictor, PredictorOutput
from medquad_qa.retrieval.tokenize import tokenize

HEURISTIC_MODEL_VERSION = "lexical-coverage-v1"

# Question-template words carry no topical content.
_TEMPLATE_WORDS = frozenset(
    tokenize(
        "what how why who when where which is are does do can could should treatment treated treat symptom symptoms "
        "sign signs cause causes caused diagnose diagnosed diagnosis prevent prevention inherited inheritance "
        "outlook prognosis risk risks information people many affected research studies being done need know "
        "genetic changes related type types",
        stem=True,
    )
)


def content_tokens(text: str) -> set[str]:
    return {t for t in tokenize(text, stem=True) if t not in _TEMPLATE_WORDS}


def coverage(question: str, evidence: str) -> float:
    q = content_tokens(question)
    if not q:
        return 0.0
    return len(q & set(tokenize(evidence, stem=True))) / len(q)


@dataclass(frozen=True)
class HeuristicGate:
    threshold: float = 1e-9
    threshold_version: str = "heuristic-unfitted"
    model_version: str = HEURISTIC_MODEL_VERSION

    @classmethod
    def from_file(cls, path: Path | None) -> HeuristicGate:
        if path is None or not path.is_file():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(threshold=float(data["threshold"]), threshold_version=str(data["threshold_version"]))

    def predict(self, question: str, evidence: list[str]) -> PredictorOutput:
        scores = [round(coverage(question, e), 6) for e in evidence]
        best = max(scores, default=0.0)
        return PredictorOutput(
            answerability_score=best,
            predicted_label=bool(evidence) and best >= self.threshold,
            threshold_version=self.threshold_version,
            model_version=self.model_version,
            per_evidence_scores=scores,
        )


def fit_threshold(scores: Sequence[float], labels: Sequence[bool]) -> tuple[float, float]:
    """Max-F1 threshold (label = score >= t). Returns (threshold, f1). Ties -> lowest threshold."""
    if len(scores) != len(labels) or not scores:
        raise ValueError("need equal-length, non-empty scores and labels")
    pos = sum(labels)
    best_t, best_f1 = 0.0, -1.0
    for t in sorted(set(scores)):
        tp = sum(1 for s, y in zip(scores, labels, strict=True) if s >= t and y)
        fp = sum(1 for s, y in zip(scores, labels, strict=True) if s >= t and not y)
        f1 = 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + (pos - tp))
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t, best_f1


def heuristic_version(threshold: float, data_sha256: str) -> str:
    payload = f"{HEURISTIC_MODEL_VERSION}|{threshold:.6f}|{data_sha256}"
    return "heuristic-" + hashlib.sha256(payload.encode()).hexdigest()[:10]


class Gate:
    """Gate wrapper: ``mode`` is 'predictor', 'heuristic' or 'off'."""

    def __init__(self, predictor: AnswerabilityPredictor | None, heuristic: HeuristicGate, mode: str = "auto") -> None:
        if mode == "auto":
            mode = "predictor" if predictor is not None else "heuristic"
        if mode == "predictor" and predictor is None:
            raise ValueError("gate mode 'predictor' requires a predictor")
        self.mode = mode
        self.predictor = predictor
        self.heuristic = heuristic

    def evaluate(self, question: str, evidence: list[str]) -> tuple[PredictorOutput | None, str, list[str]]:
        """Return (output, gate_used, warnings). Predictor failure falls back to the heuristic."""
        if self.mode == "off":
            return None, "off", []
        if self.mode == "predictor" and self.predictor is not None:
            try:
                return self.predictor.predict(question, evidence), "predictor", []
            except Exception:  # predictor bug or artifact issue: degrade, never 500 on the gate
                return self.heuristic.predict(question, evidence), "heuristic", ["answerability_fallback"]
        return self.heuristic.predict(question, evidence), "heuristic", []

    @property
    def threshold_version(self) -> str | None:
        if self.mode == "off":
            return None
        if self.mode == "predictor" and self.predictor is not None:
            return self.predictor.threshold_version
        return self.heuristic.threshold_version
