"""Rule-based refusal of personalized medical advice (all four experiment modes; D-023).

These rules are a scope policy, not a diagnosis: they decide whether a question asks for an individual
clinical decision (personal dosing, self-diagnosis, starting/stopping treatment, emergencies) instead of
general medical information. General questions about a condition or drug are NOT refused.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_RELATIVE = r"(?:child|son|daughter|baby|wife|husband|mother|father|mom|dad|partner)"
_FIRST_PERSON = rf"(?:i|i'm|im|i am|me|my|myself|mine|we|our|my {_RELATIVE})"
_FP = _FIRST_PERSON
_UP_TO_2 = r"(?:\w+\s+){0,2}"
_TREATMENT_VERBS = r"(?:take|stop|start|quit|skip|double|increase|decrease|reduce|switch|combine|mix|use|try)"
_RESULTS = r"(?:test results?|lab results?|blood test|scan|x-?ray|mri|biopsy|levels?)"

# (rule_id, pattern). Patterns are applied to a lowercased, whitespace-collapsed question.
RULES: tuple[tuple[str, str], ...] = (
    (
        "emergency",
        r"\b(?:overdos(?:e|ed|ing)|suicid\w*|kill (?:myself|me)|can'?t breathe|cannot breathe|"
        r"chest pain (?:right )?now|having a (?:heart attack|stroke)|unconscious)\b",
    ),
    (
        "personal_dosing",
        r"\b(?:how (?:much|many|often)|what dose|which dose|what dosage)\b[^?.!]*"
        rf"\b(?:should|can|do|must)\s+{_FP}\b",
    ),
    ("personal_dosing", rf"\b{_FP}\s+(?:dose|dosage)\b"),
    ("start_stop_treatment", rf"\b(?:should|can|could|may|must)\s+{_FP}\s+{_UP_TO_2}{_TREATMENT_VERBS}\b"),
    ("start_stop_treatment", rf"\bis it (?:safe|ok|okay|dangerous) for {_FP}\b"),
    ("self_diagnosis", rf"\b(?:do|does|could|might|am|is)\s+{_FP}\s+{_UP_TO_2}(?:have|has|be|got|suffer)\b"),
    ("self_diagnosis", r"\b(?:diagnose|diagnosis for) (?:me|my)\b"),
    ("self_diagnosis", rf"\bwhat(?:'s| is) wrong with {_FP}\b"),
    ("personal_management", rf"\bwhat should {_FP} (?:do|take)\b"),
    ("personal_results", rf"\b{_FP}\s+{_UP_TO_2}{_RESULTS} (?:is|are|was|were|show|came)\b"),
)

_COMPILED = tuple((rid, re.compile(p)) for rid, p in RULES)
SAFETY_RULES_VERSION = "safety-v1+" + hashlib.sha256(repr(RULES).encode()).hexdigest()[:8]

REFUSAL_MESSAGE = (
    "I can't give personal medical advice, such as whether you have a condition, what dose to take, or whether "
    "to start or stop a treatment. Please ask a qualified clinician or pharmacist. I can answer general "
    "questions about a condition or treatment."
)
EMERGENCY_MESSAGE = (
    "This may be an emergency. Please contact your local emergency number or a crisis line now. "
    "This research prototype cannot help with urgent or personal medical situations."
)


@dataclass(frozen=True, slots=True)
class SafetyDecision:
    refuse: bool
    rule_id: str | None = None

    @property
    def message(self) -> str:
        return EMERGENCY_MESSAGE if self.rule_id == "emergency" else REFUSAL_MESSAGE


def _normalize(question: str) -> str:
    q = question.lower().replace("’", "'")
    return re.sub(r"\s+", " ", q).strip()


def check_question(question: str) -> SafetyDecision:
    q = _normalize(question)
    for rule_id, pattern in _COMPILED:
        if pattern.search(q):
            return SafetyDecision(True, rule_id)
    return SafetyDecision(False, None)
