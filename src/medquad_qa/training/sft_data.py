"""Closed-book SFT examples from the approved TRAIN split (D-013).

Each example is ``[system, user=question] -> assistant=answer`` in the generator's chat template. Loss covers
only assistant tokens (answer + end-of-turn token). Answers that do not fit ``max_seq_len`` are cut at the last
sentence boundary that fits and get NO end-of-turn token, so the model never learns to stop mid-answer.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from medquad_qa.contracts.interfaces import ChatMessage
from medquad_qa.contracts.records import MedicalRecord

IGNORE_INDEX = -100
SFT_FORMAT_VERSION = "sft-closed-v1"
# Records with these flags are not used as SFT targets (shared boilerplate / non-informative answers).
EXCLUDED_FLAGS = frozenset({"boilerplate_answer", "is_boilerplate_answer", "non_informative_answer"})

_SENTENCE_END = re.compile(r"[.!?:;](?=\s)|\n")

MessageBuilder = Callable[[str], list[ChatMessage]]


def default_closed_book_messages(question: str) -> list[ChatMessage]:
    """Fallback closed-book prompt; replaced by the pipeline's frozen builder once agreed with retrieval."""
    return [
        ChatMessage(
            role="system",
            content="You provide general medical information for educational purposes. "
            "Answer the question clearly and accurately. This is not personal medical advice.",
        ),
        ChatMessage(role="user", content=question),
    ]


def read_records(path: Path) -> Iterator[MedicalRecord]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield MedicalRecord.model_validate_json(line)


@dataclass
class SFTExample:
    record_id: str
    source_name: str | None
    input_ids: list[int]
    labels: list[int]
    truncated: bool
    answer_tokens_full: int
    answer_tokens_kept: int


@dataclass
class SFTBuildReport:
    format_version: str
    max_seq_len: int
    n_records_in: int = 0
    n_examples: int = 0
    excluded: Counter[str] = field(default_factory=Counter)
    truncated_by_source: Counter[str] = field(default_factory=Counter)
    examples_by_source: Counter[str] = field(default_factory=Counter)
    answer_tokens_dropped: int = 0
    answer_tokens_total: int = 0
    input_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "max_seq_len": self.max_seq_len,
            "n_records_in": self.n_records_in,
            "n_examples": self.n_examples,
            "excluded": dict(self.excluded),
            "n_truncated": sum(self.truncated_by_source.values()),
            "truncated_by_source": dict(self.truncated_by_source),
            "examples_by_source": dict(self.examples_by_source),
            "answer_tokens_total": self.answer_tokens_total,
            "answer_tokens_dropped_by_truncation": self.answer_tokens_dropped,
            "input_sha256": self.input_sha256,
        }


def _cut_at_sentence(text: str, fits: Callable[[str], bool]) -> str:
    """Longest prefix ending at a sentence boundary that ``fits``; '' if none."""
    ends = [m.end() for m in _SENTENCE_END.finditer(text)]
    lo, hi, best = 0, len(ends) - 1, ""
    while lo <= hi:  # boundaries are monotone in length -> binary search
        mid = (lo + hi) // 2
        candidate = text[: ends[mid]].rstrip()
        if fits(candidate):
            best, lo = candidate, mid + 1
        else:
            hi = mid - 1
    return best


class SFTBuilder:
    def __init__(
        self,
        tokenizer: Any,
        *,
        max_seq_len: int = 1024,
        message_builder: MessageBuilder = default_closed_book_messages,
        end_of_turn: str = "<|im_end|>",
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.message_builder = message_builder
        self.eot_ids: list[int] = tokenizer.encode(end_of_turn, add_special_tokens=False)

    def prompt_ids(self, question: str) -> list[int]:
        msgs = [m.model_dump() for m in self.message_builder(question)]
        ids = self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=False)
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        return [int(t) for t in ids]

    def _answer_ids(self, text: str) -> list[int]:
        return [int(t) for t in self.tokenizer.encode(text, add_special_tokens=False)]

    def build_one(self, record: MedicalRecord) -> SFTExample | None:
        prompt = self.prompt_ids(record.question)
        answer = self._answer_ids(record.answer)
        budget = self.max_seq_len - len(prompt)
        if len(answer) + len(self.eot_ids) <= budget:
            target, truncated = answer + self.eot_ids, False
        else:
            kept = _cut_at_sentence(record.answer, lambda t: len(self._answer_ids(t)) <= budget)
            if not kept:
                return None
            target, truncated = self._answer_ids(kept), True
        kept_answer = len(target) - (0 if truncated else len(self.eot_ids))
        return SFTExample(
            record_id=record.record_id,
            source_name=record.source_name,
            input_ids=prompt + target,
            labels=[IGNORE_INDEX] * len(prompt) + target,
            truncated=truncated,
            answer_tokens_full=len(answer),
            answer_tokens_kept=kept_answer,
        )

    def build(self, records: Iterable[MedicalRecord]) -> tuple[list[SFTExample], SFTBuildReport]:
        report = SFTBuildReport(format_version=SFT_FORMAT_VERSION, max_seq_len=self.max_seq_len)
        examples: list[SFTExample] = []
        for rec in records:
            report.n_records_in += 1
            flags = EXCLUDED_FLAGS.intersection(rec.quality_flags)
            if flags:
                report.excluded[sorted(flags)[0]] += 1
                continue
            if not rec.answer.strip():
                report.excluded["empty_answer"] += 1
                continue
            ex = self.build_one(rec)
            if ex is None:
                report.excluded["prompt_too_long_or_no_sentence_fits"] += 1
                continue
            src = rec.source_name or "unknown"
            report.examples_by_source[src] += 1
            report.answer_tokens_total += ex.answer_tokens_full
            if ex.truncated:
                report.truncated_by_source[src] += 1
                report.answer_tokens_dropped += ex.answer_tokens_full - ex.answer_tokens_kept
            examples.append(ex)
        report.n_examples = len(examples)
        return examples, report


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_from_file(builder: SFTBuilder, path: Path) -> tuple[list[SFTExample], SFTBuildReport]:
    examples, report = builder.build(read_records(path))
    report.input_sha256 = file_sha256(path)
    return examples, report


def write_report(report: SFTBuildReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
