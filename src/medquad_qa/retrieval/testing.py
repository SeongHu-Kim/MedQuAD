"""SYNTHETIC test fixtures (not MedQuAD data). Shared by retrieval/rag tests and other teams' tests.

The texts below are invented, deliberately simple, and must never be reported as MedQuAD results.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from medquad_qa.contracts import MedicalRecord, Provenance

SYNTHETIC_CORPUS_VERSION = "synthetic-0.0.1"

# (source, topic, question, answer)
_ROWS: list[tuple[str, str, str, str]] = [
    (
        "SYN",
        "Glimmer fever",
        "What are the symptoms of Glimmer fever?",
        "Glimmer fever causes a high temperature of 39.5 C, a silver skin rash and joint pain. "
        "It does not cause hair loss.",
    ),
    (
        "SYN",
        "Glimmer fever",
        "How to treat Glimmer fever?",
        "Glimmer fever is treated with rest, fluids and the drug zorbatine 200 mg twice daily. "
        "Antibiotics are not effective against Glimmer fever.",
    ),
    (
        "SYN",
        "Glimmer fever",
        "What causes Glimmer fever?",
        "Glimmer fever is caused by the fictional glimmer virus, spread by mosquito bites.",
    ),
    (
        "SYN",
        "Quartz syndrome",
        "What are the symptoms of Quartz syndrome?",
        "Quartz syndrome causes stiff fingers, blurred vision and fatigue in adults over 40.",
    ),
    (
        "SYN",
        "Quartz syndrome",
        "Is Quartz syndrome inherited?",
        "Quartz syndrome is inherited in an autosomal recessive pattern. Carriers have no symptoms.",
    ),
    (
        "SYN2",
        "Velvet cough",
        "What is Velvet cough?",
        "Velvet cough is a mild airway irritation without fever. It usually resolves within 10 days.",
    ),
    (
        "SYN2",
        "Velvet cough",
        "How to prevent Velvet cough?",
        "Velvet cough can be prevented by avoiding dust and washing hands. There is no vaccine.",
    ),
    # duplicate answer of the first record under another source (same duplicate group)
    (
        "SYN2",
        "Glimmer fever",
        "What are the signs of Glimmer fever?",
        "Glimmer fever causes a high temperature of 39.5 C, a silver skin rash and joint pain. "
        "It does not cause hair loss.",
    ),
]

LONG_ANSWER_WORDS = 450


def _rid(*parts: str) -> str:
    return "mq-" + hashlib.sha256("\x1f".join(("synthetic",) + parts).encode()).hexdigest()[:16]


def synthetic_records(*, with_long: bool = True) -> list[MedicalRecord]:
    """Small deterministic corpus. With ``with_long`` adds one 450-word answer to exercise chunking."""
    rows = list(_ROWS)
    if with_long:
        words = [f"lorem{i % 37}" for i in range(LONG_ANSWER_WORDS)]
        words[300:303] = ["unique", "marker", "zephyrine"]
        rows.append(("SYN", "Long topic", "What is the long topic?", " ".join(words)))
    prov_sha = "0" * 64
    out = []
    for i, (src, topic, q, a) in enumerate(rows):
        rid = _rid(src, topic, q, a)
        dup = f"dg-{hashlib.sha256(a.encode()).hexdigest()[:12]}"
        out.append(
            MedicalRecord(
                record_id=rid,
                question=q,
                answer=a,
                question_raw=q,
                answer_raw=a,
                source_name=src,
                topic=topic,
                provenance=Provenance(
                    dataset_name="SYNTHETIC test fixture",
                    source_file="tests/fixtures/synthetic",
                    source_file_sha256=prov_sha,
                    row_index=i,
                    pipeline_version="synthetic",
                ),
                content_hash=hashlib.sha256(f"{q}\x1f{a}".encode()).hexdigest(),
                duplicate_group_id=dup,
                split_group_id=f"sg-{topic.lower().replace(' ', '-')}",
                quality_flags=[],
            )
        )
    return out


def write_synthetic_corpus(directory: Path, *, with_long: bool = True) -> tuple[Path, Path]:
    """Write corpus.jsonl and corpus_manifest.json; return (corpus_path, manifest_path)."""
    directory.mkdir(parents=True, exist_ok=True)
    corpus = directory / "corpus.jsonl"
    manifest = directory / "corpus_manifest.json"
    with corpus.open("w", encoding="utf-8") as fh:
        for rec in synthetic_records(with_long=with_long):
            fh.write(rec.model_dump_json() + "\n")
    manifest.write_text(json.dumps({"corpus_version": SYNTHETIC_CORPUS_VERSION}) + "\n", encoding="utf-8")
    return corpus, manifest
