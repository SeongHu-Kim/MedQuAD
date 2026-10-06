"""Keras answerability model on tiny synthetic pairs (skipped when TensorFlow is not installed).

Runs in a subprocess: importing TensorFlow before Triton (pulled in by torch._dynamo / PEFT training) segfaults
this environment (`python -c "import tensorflow; import triton"` -> SIGSEGV), so TF must never share a process
with the torch training tests.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("tensorflow") is None, reason="tensorflow not installed")

SCRIPT = r"""
import sys
from pathlib import Path
sys.path.insert(0, {tests_dir!r})
from test_training_answerability import _records
from medquad_qa.contracts.interfaces import AnswerabilityPredictor
from medquad_qa.training.answerability_pairs import build_pairs
from medquad_qa.training.keras_answerability import KerasAnswerabilityPredictor, KerasConfig, train_keras

class TinyConfig(KerasConfig):
    vocab_size = 500
    question_len = 8
    evidence_len = 32
    embed_dim = 8
    gru_units = 4
    dense_units = 8
    batch_size = 16
    max_epochs = 2

out = Path({out!r})
pairs = {{
    "train": build_pairs(_records("tr", 12, 0), "train"),
    "validation": build_pairs(_records("va", 6, 1000), "validation"),
    "test": build_pairs(_records("te", 6, 2000), "test"),
}}
result = train_keras(pairs, out, TinyConfig)
assert result["epochs_run"] >= 1 and "test" in result
assert result["threshold"]["threshold_rule"] == "max_f1_on_validation"
pred = KerasAnswerabilityPredictor(out)
assert isinstance(pred, AnswerabilityPredictor)
o = pred.predict(pairs["validation"][0].question, ["unrelated", pairs["validation"][0].evidence_text])
assert len(o.per_evidence_scores) == 2 and o.answerability_score == max(o.per_evidence_scores)
assert pred.predict("q?", []).answerability_score == 0.0
print("KERAS_OK")
"""


def test_keras_trains_saves_and_predicts(tmp_path: Path) -> None:
    code = SCRIPT.format(tests_dir=str(Path(__file__).parent), out=str(tmp_path / "keras"))
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)  # noqa: S603
    assert proc.returncode == 0 and "KERAS_OK" in proc.stdout, proc.stderr[-3000:]
