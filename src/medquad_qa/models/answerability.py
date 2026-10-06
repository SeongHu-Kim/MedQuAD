"""Answerability predictors (non-clinical): does the evidence text suffice to answer the question?

``LexicalFeaturizer`` + logistic regression is the statistical baseline AND the serving predictor: it is stored
as plain JSON (IDF table, feature standardisation, coefficients) and scored with numpy only, so the API needs
neither scikit-learn pickles nor TensorFlow (D-014).
The Keras model lives in ``medquad_qa.training.keras_answerability``.
Scores are answerability estimates for retrieved text; they are NOT diagnostic or medical confidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from medquad_qa.contracts.answerability import PredictorOutput
from medquad_qa.contracts.interfaces import ArtifactUnavailableError

_TOKEN = re.compile(r"[a-z0-9]+")
# fmt: off
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "has", "have", "how",
    "i", "if", "in", "into", "is", "it", "its", "may", "me", "my", "of", "on", "or", "should", "so", "than",
    "that", "the", "their", "them", "there", "these", "they", "this", "to", "was", "what", "when", "where",
    "which", "who", "why", "will", "with", "you", "your", "about", "after", "all", "also", "any", "been", "being",
    "both", "but", "each", "more", "most", "no", "not", "other", "out", "over", "such", "some", "very", "would",
    "could", "people", "person", "get"
})
# fmt: on
FEATURE_NAMES = (
    "q_coverage",
    "q_coverage_idf",
    "jaccard",
    "q_bigram_coverage",
    "tfidf_cosine",
    "q_coverage_lead30",
    "log_len_evidence",
    "log_len_question",
)


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def content(toks: Iterable[str]) -> list[str]:
    return [t for t in toks if t not in STOPWORDS and len(t) > 1]


class LexicalFeaturizer:
    """Question/evidence overlap features with an IDF table fitted on TRAIN evidence only."""

    def __init__(self, idf: dict[str, float], default_idf: float) -> None:
        self.idf = idf
        self.default_idf = default_idf

    @classmethod
    def fit(cls, evidence_texts: Iterable[str]) -> LexicalFeaturizer:
        df: Counter[str] = Counter()
        n = 0
        for text in evidence_texts:
            n += 1
            df.update(set(content(tokens(text))))
        idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items() if c >= 2}
        return cls(idf, math.log(1 + n) + 1.0)

    def _w(self, t: str) -> float:
        return self.idf.get(t, self.default_idf)

    def features(self, question: str, evidence: str) -> list[float]:
        q_all, e_all = tokens(question), tokens(evidence)
        q, e = content(q_all), content(e_all)
        qs, es = set(q), set(e)
        lead = set(content(e_all[:30]))
        cov = len(qs & es) / len(qs) if qs else 0.0
        wq = sum(self._w(t) for t in qs)
        cov_idf = sum(self._w(t) for t in qs & es) / wq if wq else 0.0
        jac = len(qs & es) / len(qs | es) if qs | es else 0.0
        qb = set(zip(q, q[1:], strict=False))
        eb = set(zip(e, e[1:], strict=False))
        bcov = len(qb & eb) / len(qb) if qb else 0.0
        qv, ev = Counter(q), Counter(e)
        dot = sum(qv[t] * ev[t] * self._w(t) ** 2 for t in qs & es)
        nq = math.sqrt(sum((c * self._w(t)) ** 2 for t, c in qv.items()))
        ne = math.sqrt(sum((c * self._w(t)) ** 2 for t, c in ev.items()))
        cos = dot / (nq * ne) if nq and ne else 0.0
        lead_cov = len(qs & lead) / len(qs) if qs else 0.0
        return [cov, cov_idf, jac, bcov, cos, lead_cov, math.log1p(len(e_all)), math.log1p(len(q_all))]

    def matrix(self, questions: Sequence[str], evidence: Sequence[str]) -> np.ndarray:
        return np.asarray([self.features(q, e) for q, e in zip(questions, evidence, strict=True)], dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {"idf": self.idf, "default_idf": self.default_idf}


class LexicalLRModel:
    """Standardised features -> logistic regression, serialised as JSON."""

    def __init__(
        self, featurizer: LexicalFeaturizer, mean: np.ndarray, std: np.ndarray, coef: np.ndarray, intercept: float
    ) -> None:
        self.featurizer = featurizer
        self.mean, self.std, self.coef, self.intercept = mean, std, coef, intercept

    def score_matrix(self, x: np.ndarray) -> np.ndarray:
        z = ((x - self.mean) / self.std) @ self.coef + self.intercept
        return np.asarray(1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0))))

    def score(self, questions: Sequence[str], evidence: Sequence[str]) -> np.ndarray:
        if not questions:
            return np.zeros(0)
        return self.score_matrix(self.featurizer.matrix(questions, evidence))

    def to_json(self) -> str:
        return json.dumps(
            {
                "kind": "lexical_lr",
                "feature_names": list(FEATURE_NAMES),
                "mean": self.mean.tolist(),
                "std": self.std.tolist(),
                "coef": self.coef.tolist(),
                "intercept": self.intercept,
                "featurizer": self.featurizer.to_dict(),
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str) -> LexicalLRModel:
        d = json.loads(text)
        if d.get("feature_names") != list(FEATURE_NAMES):
            raise ArtifactUnavailableError("answerability model feature set does not match this code version")
        f = d["featurizer"]
        return cls(
            LexicalFeaturizer(f["idf"], f["default_idf"]),
            np.asarray(d["mean"]),
            np.asarray(d["std"]),
            np.asarray(d["coef"]),
            float(d["intercept"]),
        )


def lexlr_version(model_json: bytes) -> str:
    return f"answerability-lexlr@{hashlib.sha256(model_json).hexdigest()[:12]}"


class LexicalAnswerabilityPredictor:
    """``AnswerabilityPredictor`` backed by the JSON LR model and a validation-chosen threshold (D-018).

    Directory layout: ``model.json`` and ``threshold.json`` (``{"threshold": float, "threshold_version": str}``).
    """

    def __init__(self, model: LexicalLRModel, threshold: float, model_version: str, threshold_version: str) -> None:
        self.model = model
        self.threshold = threshold
        self.model_version = model_version
        self.threshold_version = threshold_version

    @classmethod
    def load(cls, directory: Path) -> LexicalAnswerabilityPredictor:
        model_path, thr_path = directory / "model.json", directory / "threshold.json"
        if not model_path.is_file() or not thr_path.is_file():
            raise ArtifactUnavailableError(f"answerability model not found in {directory}")
        raw = model_path.read_text(encoding="utf-8")
        thr = json.loads(thr_path.read_text(encoding="utf-8"))
        version = lexlr_version(raw.encode("utf-8"))
        return cls(LexicalLRModel.from_json(raw), float(thr["threshold"]), version, str(thr["threshold_version"]))

    def predict(self, question: str, evidence: list[str]) -> PredictorOutput:
        if not evidence:
            return PredictorOutput(
                answerability_score=0.0,
                predicted_label=False,
                threshold_version=self.threshold_version,
                model_version=self.model_version,
            )
        scores = [float(s) for s in self.model.score([question] * len(evidence), evidence)]
        best = max(scores)
        return PredictorOutput(
            answerability_score=best,
            predicted_label=best >= self.threshold,
            threshold_version=self.threshold_version,
            model_version=self.model_version,
            per_evidence_scores=scores,
        )
