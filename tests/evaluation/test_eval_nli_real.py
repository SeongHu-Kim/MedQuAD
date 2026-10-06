"""Real-model smoke test of the pinned NLI support scorer (skipped unless the checkpoint is cached locally)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medquad_qa.evaluation.support import NLISupportScorer

CFG = json.loads((Path(__file__).parents[2] / "configs/evaluation/metrics.json").read_text())["support_scorer"]


@pytest.mark.real_model
def test_nli_scorer_entailment_vs_contradiction() -> None:
    scorer = NLISupportScorer(CFG["model_id"], CFG["revision"])
    try:
        scorer._load()
    except OSError as exc:  # not in the local HF cache: offline run
        pytest.skip(f"NLI checkpoint not cached: {exc}")
    evidence = "Synthetic condition X is caused by a deficiency of vitamin Q. It is not contagious."
    entailed, contradicted, unrelated = scorer.score_many(
        [
            (evidence, "Condition X is caused by low vitamin Q."),
            (evidence, "Condition X is contagious."),
            (evidence, "Condition X is treated with surgery."),
        ]
    )
    assert entailed > 0.8
    assert contradicted < 0.2 and unrelated < 0.5
    long_evidence = " ".join(["Filler sentence about nothing in particular."] * 200) + " " + evidence
    assert len(scorer.windows(long_evidence)) > 1
    assert scorer.score(long_evidence, "Condition X is caused by low vitamin Q.") > 0.8
