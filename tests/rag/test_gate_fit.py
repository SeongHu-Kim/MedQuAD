from __future__ import annotations

import json
from pathlib import Path

from medquad_qa.contracts import AnswerabilityPair
from medquad_qa.rag.gate import HeuristicGate
from medquad_qa.rag.gate_fit import main


def _pair(i: int, q: str, ev: str, label: bool, split: str = "validation") -> str:
    return AnswerabilityPair(
        pair_id=f"p{i}",
        question=q,
        question_record_id="mq-" + "a" * 16,
        evidence_record_id="mq-" + "b" * 16,
        evidence_text=ev,
        label=label,
        negative_type=None if label else "easy_random",
        pair_rule_version="t",
        split=split,
        split_group_id="g",  # type: ignore[arg-type]
    ).model_dump_json()


def test_fit_writes_threshold_and_gate_loads_it(tmp_path: Path) -> None:
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(
        "\n".join(
            [
                _pair(0, "What causes Glimmer fever?", "Glimmer fever is caused by a virus.", True),
                _pair(1, "What causes Glimmer fever?", "Velvet cough is mild.", False),
                _pair(2, "Is Quartz syndrome inherited?", "Quartz syndrome is autosomal recessive.", True),
                _pair(3, "Is Quartz syndrome inherited?", "Glimmer fever rash.", False),
            ]
        )
        + "\n"
    )
    out = tmp_path / "gate.json"
    assert main(["--pairs", str(pairs), "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["validation_f1"] == 1.0 and data["n_pairs"] == 4
    gate = HeuristicGate.from_file(out)
    assert gate.threshold_version == data["threshold_version"]
    assert gate.predict("What causes Glimmer fever?", ["Glimmer fever is caused by a virus."]).predicted_label
    assert not gate.predict("What causes Glimmer fever?", ["Velvet cough is mild."]).predicted_label
    assert not gate.predict("What causes Glimmer fever?", []).predicted_label


def test_fit_rejects_non_validation_rows(tmp_path: Path) -> None:
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(_pair(0, "q?", "e", True, split="test") + "\n")
    assert main(["--pairs", str(pairs), "--out", str(tmp_path / "g.json")]) == 2
    assert not (tmp_path / "g.json").exists()
