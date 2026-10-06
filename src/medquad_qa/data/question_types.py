"""Rule-based question-type labelling from question templates (configs/data/question_types.yaml)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from medquad_qa.data.config import DEFAULT_QTYPE_CONFIG


@dataclass(frozen=True)
class QuestionTypeRules:
    rule_version: str
    default_type: str
    rules: tuple[tuple[str, re.Pattern[str]], ...]

    def classify(self, question: str) -> str:
        for qtype, pattern in self.rules:
            if pattern.search(question):
                return qtype
        return self.default_type

    def table(self) -> list[dict[str, str]]:
        return [{"type": t, "pattern": p.pattern} for t, p in self.rules]


def load_rules(path: Path = DEFAULT_QTYPE_CONFIG) -> QuestionTypeRules:
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    rules = tuple((r["type"], re.compile(r["pattern"], re.IGNORECASE)) for r in raw["rules"])
    return QuestionTypeRules(raw["rule_version"], raw["default_type"], rules)
