"""Example questions for the GUI and a behaviour-only check of a response against the intended behaviour (D-069).

Owner: service-platform-engineer (written by the lead under the D-069 ownership exception). Pure functions,
tested offline. The check compares observable response facts only (answered or abstained, abstention reason,
crisis message, citations present); it is not an accuracy judgement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from medquad_qa.rag.safety import EMERGENCY_MESSAGE
from medquad_qa.rag.safety_check import SAFETY_CHECK_FAILED

EXAMPLES_PATH = Path(__file__).resolve().parents[3] / "configs" / "service" / "example_questions.yaml"
MODES = ("base", "rag", "finetuned", "finetuned_rag")

BEHAVIOURS: dict[str, str] = {
    "answered_uncited": "인용 없이 답변",
    "answered_cited": "근거를 인용해 답변",
    "answered": "답변 (거부 아님)",
    "abstain_no_evidence": "근거 없음/부족으로 거부",
    "refuse_personal": "개인 의료 조언 거부",
    "crisis_message": "위기 안내 메시지",
    "abstain_missing_citations": "인용 누락으로 거부",
}
_NO_EVIDENCE = frozenset({"no_relevant_evidence", "insufficient_evidence"})


@dataclass(frozen=True)
class Example:
    group: str
    question: str
    expected: dict[str, str]
    expected_text: str
    note: str | None = None


@dataclass(frozen=True)
class Check:
    expected_code: str | None
    expected_label: str
    actual_label: str
    match: bool | None  # None: no intended behaviour for this mode
    details: list[str] = field(default_factory=list)


def load_examples(path: Path = EXAMPLES_PATH) -> list[tuple[str, list[Example]]]:
    """Load and validate the example config; returns [(group name, examples)] in file order."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError(f"{path}: expected a mapping with version: 1")
    groups: list[tuple[str, list[Example]]] = []
    seen: set[str] = set()
    for g in data.get("groups") or []:
        name = str(g["name"])
        items: list[Example] = []
        for e in g.get("examples") or []:
            question = str(e["question"]).strip()
            if not question or question in seen:
                raise ValueError(f"{path}: empty or duplicate question in group {name!r}")
            seen.add(question)
            expected = {str(k): str(v) for k, v in (e.get("expected") or {}).items()}
            if not expected:
                raise ValueError(f"{path}: no expected behaviour for {question!r}")
            for mode, code in expected.items():
                if mode != "all" and mode not in MODES:
                    raise ValueError(f"{path}: unknown mode {mode!r} for {question!r}")
                if code not in BEHAVIOURS:
                    raise ValueError(f"{path}: unknown behaviour {code!r} for {question!r}")
            items.append(Example(name, question, expected, str(e["expected_text"]), e.get("note") or None))
        groups.append((name, items))
    return groups


def find_example(groups: list[tuple[str, list[Example]]], question: str) -> Example | None:
    """The example whose question equals ``question`` exactly (after stripping), else None."""
    q = question.strip()
    return next((e for _, items in groups for e in items if e.question == q), None)


def expected_for(example: Example, mode: str) -> str | None:
    return example.expected.get(mode, example.expected.get("all"))


def actual_state(resp: dict[str, Any]) -> tuple[str, str]:
    """(state code, Korean label) of a QAResponse JSON, from observable fields only."""
    n_cit = len(resp.get("citations") or [])
    if not resp.get("abstained"):
        return ("answered_cited", f"답변 (인용 {n_cit}건)") if n_cit else ("answered_uncited", "답변 (인용 없음)")
    reason = resp.get("abstention_reason") or "unspecified"
    if reason == "personalized_medical_advice":
        if SAFETY_CHECK_FAILED in (resp.get("warnings") or []):
            return "safety_check_failed", "안전 검사 실패로 위기 안내"
        if (resp.get("answer") or "").strip() == EMERGENCY_MESSAGE.strip():
            return "crisis_message", "위기 안내 메시지"
        return "refuse_personal", "개인 의료 조언 거부"
    if reason in _NO_EVIDENCE:
        return "abstain_no_evidence", f"근거 없음/부족으로 거부 ({reason})"
    if reason == "missing_citations":
        return "abstain_missing_citations", "인용 누락으로 거부 (missing_citations)"
    return f"abstain:{reason}", f"거부 ({reason})"


def _matches(code: str, state: str) -> bool:
    if code == "answered":
        return state in ("answered_cited", "answered_uncited")
    if code == "refuse_personal":
        return state in ("refuse_personal", "crisis_message")
    return code == state


def compare(example: Example, mode: str, resp: dict[str, Any]) -> Check:
    """Intended vs actual behaviour for one response. Behaviour check only; not an accuracy judgement."""
    state, label = actual_state(resp)
    code = expected_for(example, mode)
    details: list[str] = []
    if state == "crisis_message" and code == "refuse_personal":
        details.append("개인 조언 거부가 위기 안내 메시지 형태로 표시됨")
    if code is None:
        return Check(None, "이 모드에 대한 기대 동작 없음 (비교하지 않음)", label, None, details)
    return Check(code, BEHAVIOURS[code], label, _matches(code, state), details)
