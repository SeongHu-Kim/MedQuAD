"""Keras answerability model on tiny synthetic pairs (skipped when TensorFlow is not installed)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("tensorflow")

from test_training_answerability import _records  # noqa: E402

from medquad_qa.contracts.interfaces import AnswerabilityPredictor  # noqa: E402
from medquad_qa.training.answerability_pairs import build_pairs  # noqa: E402
from medquad_qa.training.keras_answerability import KerasAnswerabilityPredictor, KerasConfig, train_keras  # noqa: E402


class TinyConfig(KerasConfig):
    vocab_size = 500
    question_len = 8
    evidence_len = 32
    embed_dim = 8
    gru_units = 4
    dense_units = 8
    batch_size = 16
    max_epochs = 2


def test_keras_trains_saves_and_predicts(tmp_path: Path) -> None:
    pairs = {
        "train": build_pairs(_records("tr", 12, 0), "train"),
        "validation": build_pairs(_records("va", 6, 1000), "validation"),
        "test": build_pairs(_records("te", 6, 2000), "test"),
    }
    result = train_keras(pairs, tmp_path / "keras", TinyConfig)
    assert result["epochs_run"] >= 1 and "test" in result
    assert result["threshold"]["threshold_rule"] == "max_f1_on_validation"

    pred = KerasAnswerabilityPredictor(tmp_path / "keras")
    assert isinstance(pred, AnswerabilityPredictor)
    out = pred.predict(pairs["validation"][0].question, ["unrelated", pairs["validation"][0].evidence_text])
    assert len(out.per_evidence_scores) == 2 and out.answerability_score == max(out.per_evidence_scores)
    assert pred.predict("q?", []).answerability_score == 0.0
