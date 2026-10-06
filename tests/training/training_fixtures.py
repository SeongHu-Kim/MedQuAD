"""Synthetic (non-MedQuAD) records and exports for training tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

from medquad_qa.contracts.records import MedicalRecord, Provenance

TOPICS = ["Anemia", "Asthma", "Gout", "Migraine"]
QTYPES = {"symptoms": "What are the symptoms of {t}?", "treatment": "How is {t} treated?"}


def synthetic_record(
    i: int, topic: str, qtype: str, answer: str, group: str, flags: list[str] | None = None
) -> MedicalRecord:
    question = QTYPES[qtype].format(t=topic)
    rid = "mq-" + hashlib.sha256(f"syn-{i}".encode()).hexdigest()[:16]
    return MedicalRecord(
        record_id=rid,
        question=question,
        answer=answer,
        question_raw=question,
        answer_raw=answer,
        source_name="SYN",
        topic=topic,
        question_type=qtype,
        provenance=Provenance(
            dataset_name="synthetic fixture",
            source_file="tests/fixture",
            source_file_sha256="0" * 64,
            row_index=i,
            pipeline_version="test",
        ),
        content_hash=hashlib.sha256(f"{question}\x1f{answer}".encode()).hexdigest(),
        duplicate_group_id=f"dg-{i}",
        split_group_id=group,
        quality_flags=flags or [],
    )


def make_split(split: str, start: int) -> list[MedicalRecord]:
    out = []
    i = start
    for t in TOPICS:
        for q in QTYPES:
            answer = f"{t} {q} information sentence one. It has a second sentence. And a third one for length."
            out.append(synthetic_record(i, f"{t}-{split}", q, answer, group=f"g-{split}-{t}"))
            i += 1
    return out


def write_jsonl(records: list[MedicalRecord], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8")
    return path
