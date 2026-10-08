"""Integration: fresh-retest harness wiring against the safety-v4 interface (fake base generator; synthetic text)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from medquad_qa.contracts import EXPERIMENT_MODES, ChatMessage, QARequest
from medquad_qa.evaluation import fresh_retest as fr
from medquad_qa.evaluation.fakes import ScriptedGenerator
from medquad_qa.evaluation.fixture_retriever import FixtureRetriever

pytestmark = pytest.mark.integration
safety_check = pytest.importorskip("medquad_qa.rag.safety_check")

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("run_fresh_retest", ROOT / "scripts/evaluation/run_fresh_retest.py")
assert spec and spec.loader
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)

Q_GENERAL = "I read about synthetic condition Q yesterday."  # reaches the model stage (not safe harbour, no rule)
EVIDENCE = "Synthetic condition Q causes mild fever and tiredness."


def fake(label: str) -> ScriptedGenerator:
    def reply(messages: list[ChatMessage]) -> str:
        if messages and messages[0].content.startswith(safety_check.CHECK_SYSTEM_PROMPT):
            return label
        return "Synthetic condition Q causes mild fever and tiredness [E1]."

    return ScriptedGenerator(reply, model_version="fake-base@0")


@pytest.mark.parametrize(
    "label,refuse,emergency", [("GENERAL", False, False), ("CRISIS", True, True), ("garbled", True, True)]
)
def test_harness_checker_and_four_mode_agreement(label: str, refuse: bool, emergency: bool) -> None:
    from medquad_qa.rag.factory import build_pipeline
    from medquad_qa.rag.safety import EMERGENCY_MESSAGE

    gen = fake(label)
    pipe = build_pipeline(retriever=FixtureRetriever({Q_GENERAL: [EVIDENCE]}), generator_provider=lambda _v: gen)
    checker = harness.make_checker(pipe)
    assert checker is pipe.safety_checker  # the same object the pipeline uses
    o = fr.outcome_from_decision(harness.decide(Q_GENERAL, checker))
    assert o.refuse is refuse and o.stage == "model"
    if label == "garbled":
        assert o.failed_check and o.rule_id == fr.SAFETY_CHECK_FAILED
    for mode in EXPERIMENT_MODES:
        r = pipe.answer(QARequest(question=Q_GENERAL, experiment_mode=mode, top_k=1), f"w-{mode}")
        assert fr.agreement(o, r, EMERGENCY_MESSAGE), (label, mode, r.abstention_reason)
        if emergency:
            assert r.answer == EMERGENCY_MESSAGE
