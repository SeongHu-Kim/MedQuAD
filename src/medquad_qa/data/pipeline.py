"""End-to-end data build: raw CSV -> corpus, exclusions, groups, splits, leakage report, exports.

Everything is computed in memory and returned as ``{repo-relative path: bytes}`` so a rebuild can be
compared byte-for-byte with the files on disk. No timestamps or absolute paths enter any output.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from medquad_qa.contracts import CONTRACTS_VERSION, ContractViolationError, MedicalRecord, Provenance
from medquad_qa.data import audit as audit_mod
from medquad_qa.data.config import BuildConfig
from medquad_qa.data.grouping import GroupInput, build_groups
from medquad_qa.data.ids import content_hash, record_id
from medquad_qa.data.ingest import RawRow, file_sha256, read_rows
from medquad_qa.data.leakage import leakage_report
from medquad_qa.data.manifests import json_bytes, jsonl_bytes, sha256_bytes
from medquad_qa.data.normalize import normalize_question, normalize_text, normalize_topic, topic_key
from medquad_qa.data.quality import (
    EXCLUDE_DUPLICATE,
    EXCLUDE_EMPTY,
    EXCLUDE_NON_INFORMATIVE,
    EXCLUSION_REASONS,
    FLAG_BOILERPLATE,
    non_informative_kind,
    record_flags,
)
from medquad_qa.data.question_types import QuestionTypeRules
from medquad_qa.data.splits import SPLITS, assign_splits

LICENCE_NOTE = (
    "Kaggle owner/slug and licence not verifiable from the CSV export. Local research use only; "
    "do not redistribute the dataset or its text."
)
GROUPING_DISCLOSURE = (
    "The export has no source URLs or document IDs, so split groups are derived from content and topic "
    "(weaker than document-level grouping). Leakage is reduced, not proven absent."
)


@dataclass(frozen=True)
class Exclusion:
    row_index: int
    reason: str
    detail: str | None
    source: str
    topic: str | None
    canonical_row_index: int | None = None


@dataclass
class BuildResult:
    records: list[MedicalRecord]
    splits: list[str]  # aligned with records
    exclusions: list[Exclusion]
    corpus_version: str
    split_version: str
    leakage: dict[str, Any]
    files: dict[str, bytes] = field(default_factory=dict)
    file_groups: dict[str, list[str]] = field(default_factory=dict)


def _select_rows(rows: list[RawRow], cfg: BuildConfig) -> tuple[list[RawRow], dict[int, list[int]], list[Exclusion]]:
    """Apply exclusions in reason order: empty answer, exact duplicate row, non-informative answer."""
    kept: list[RawRow] = []
    first_row: dict[tuple[str, str, str, str], int] = {}
    dup_rows: dict[int, list[int]] = defaultdict(list)
    excluded: list[Exclusion] = []
    for r in rows:
        answer = normalize_text(r.answer)
        topic = normalize_topic(r.focus_area)
        key = (r.source, r.focus_area, r.question, r.answer)
        if not answer:
            excluded.append(Exclusion(r.row_index, EXCLUDE_EMPTY, None, r.source, topic))
        elif key in first_row:
            canon = first_row[key]
            dup_rows[canon].append(r.row_index)
            excluded.append(Exclusion(r.row_index, EXCLUDE_DUPLICATE, None, r.source, topic, canon))
        elif (kind := non_informative_kind(answer, cfg.exclusion)) is not None:
            excluded.append(Exclusion(r.row_index, EXCLUDE_NON_INFORMATIVE, kind, r.source, topic))
        else:
            first_row[key] = r.row_index
            kept.append(r)
    return kept, dup_rows, excluded


def run_build(source_path: Path, cfg: BuildConfig, qrules: QuestionTypeRules) -> BuildResult:
    src_sha = file_sha256(source_path)
    if cfg.source.expected_sha256 and src_sha != cfg.source.expected_sha256:
        raise ContractViolationError(f"source sha256 {src_sha} != expected {cfg.source.expected_sha256}")
    rows = read_rows(source_path, cfg.source.expected_columns)
    kept, dup_rows, excluded = _select_rows(rows, cfg)

    reconciled = len(kept) + len(excluded)
    if reconciled != len(rows):
        raise ContractViolationError(f"row reconciliation failed: {len(kept)} + {len(excluded)} != {len(rows)}")

    questions = [normalize_question(raw.question) for raw in kept]
    answers = [normalize_text(raw.answer) for raw in kept]
    topics = [normalize_topic(raw.focus_area) for raw in kept]
    rids = [record_id(raw.source, raw.focus_area, raw.question, raw.answer) for raw in kept]
    if len(set(rids)) != len(rids):
        raise ContractViolationError("record_id collision after duplicate collapse")
    rid_of_row = {raw.row_index: rid for raw, rid in zip(kept, rids, strict=True)}

    grouping = build_groups(
        [GroupInput(rid, q, a, t) for rid, q, a, t in zip(rids, questions, answers, topics, strict=True)],
        cfg.grouping,
    )

    records: list[MedicalRecord] = []
    for i, raw in enumerate(kept):
        flags = record_flags(questions[i], answers[i], topics[i], len(dup_rows.get(raw.row_index, [])), cfg.quality)
        if grouping.boilerplate[i]:
            flags.insert(0, FLAG_BOILERPLATE)
        records.append(
            MedicalRecord(
                record_id=rids[i],
                question=questions[i],
                answer=answers[i],
                question_raw=raw.question,
                answer_raw=raw.answer,
                source_name=raw.source or None,
                topic=topics[i],
                question_type=qrules.classify(questions[i]),
                provenance=Provenance(
                    dataset_name=cfg.source.dataset_name,
                    source_file=cfg.source.path,
                    source_file_sha256=src_sha,
                    row_index=raw.row_index,
                    kaggle_dataset_ref=cfg.source.kaggle_dataset_ref,
                    pipeline_version=cfg.pipeline_version,
                    duplicate_row_indices=dup_rows.get(raw.row_index, []),
                ),
                content_hash=content_hash(questions[i], answers[i]),
                duplicate_group_id=grouping.duplicate_group_ids[i],
                split_group_id=grouping.split_group_ids[i],
                quality_flags=sorted(flags),
            )
        )

    proc = cfg.outputs.processed_dir
    man = cfg.outputs.manifests_dir
    files: dict[str, bytes] = {}

    corpus_bytes = jsonl_bytes(r.model_dump(mode="json") for r in records)
    corpus_sha = sha256_bytes(corpus_bytes)
    corpus_version = f"medquad-{cfg.corpus_semver}-{corpus_sha[:12]}"
    files[f"{proc}/corpus.jsonl"] = corpus_bytes

    excl_rows = [
        {
            "row_index": e.row_index,
            "reason": e.reason,
            "detail": e.detail,
            "source": e.source,
            "topic": e.topic,
            "canonical_row_index": e.canonical_row_index,
            "canonical_record_id": rid_of_row.get(e.canonical_row_index) if e.canonical_row_index is not None else None,
        }
        for e in excluded
    ]
    files[f"{man}/exclusions.jsonl"] = jsonl_bytes(excl_rows)
    excl_counts = Counter(e.reason for e in excluded)
    excl_detail = Counter(f"{e.reason}:{e.detail}" for e in excluded if e.detail)

    files[f"{man}/source_dataset.json"] = json_bytes(
        {
            "dataset_name": cfg.source.dataset_name,
            "source_file": cfg.source.path,
            "sha256": src_sha,
            "size_bytes": source_path.stat().st_size,
            "rows": len(rows),
            "columns": cfg.source.expected_columns,
            "kaggle_dataset_ref": cfg.source.kaggle_dataset_ref,
            "licence": LICENCE_NOTE,
            "has_source_urls": False,
            "has_document_ids": False,
        }
    )
    files[f"{man}/corpus_manifest.json"] = json_bytes(
        {
            "corpus_version": corpus_version,
            "corpus_path": f"{proc}/corpus.jsonl",
            "corpus_sha256": corpus_sha,
            "records": len(records),
            "records_by_source": dict(sorted(Counter(r.source_name for r in records if r.source_name).items())),
            "source_rows": len(rows),
            "excluded_rows": len(excluded),
            "excluded_by_reason": {k: excl_counts.get(k, 0) for k in EXCLUSION_REASONS},
            "excluded_by_detail": dict(sorted(excl_detail.items())),
            "reconciliation": f"{len(records)} + {len(excluded)} = {len(rows)}",
            "record_id_rule": "mq- + sha256('medquad-rid-v1' US source US focus_area_raw US question_raw US "
            "answer_raw)[:16]",
            "normalization": "NFC; newline unification; per-line whitespace collapse; blank-line collapse; "
            "questions also ' ?' -> '?'. No lowercasing or punctuation/number/negation edits.",
            "pipeline_version": cfg.pipeline_version,
            "contracts_version": CONTRACTS_VERSION,
            "source_sha256": src_sha,
            "grouping_disclosure": GROUPING_DISCLOSURE,
        }
    )

    # ---- splits
    group_split, dominant = assign_splits(
        [r.split_group_id for r in records], [r.source_name or "" for r in records], cfg.split.ratios, cfg.split.seed
    )
    splits = [group_split[r.split_group_id] for r in records]
    manifest_rows = [
        {
            "record_id": r.record_id,
            "split": s,
            "split_group_id": r.split_group_id,
            "duplicate_group_id": r.duplicate_group_id,
            "source": r.source_name,
            "topic": r.topic,
            "question_type": r.question_type,
            "quality_flags": r.quality_flags,
        }
        for r, s in zip(records, splits, strict=True)
    ]
    split_bytes = jsonl_bytes(manifest_rows)
    split_version = f"split-{cfg.split.seed}-{sha256_bytes(split_bytes)[:12]}"
    files[f"{man}/split_manifest.jsonl"] = split_bytes

    rows_per = Counter(splits)
    groups_per: dict[str, set[str]] = defaultdict(set)
    rows_by_src: dict[str, Counter[str]] = defaultdict(Counter)
    groups_by_dom: dict[str, Counter[str]] = defaultdict(Counter)
    qtype_per: dict[str, Counter[str | None]] = defaultdict(Counter)
    for r, s in zip(records, splits, strict=True):
        groups_per[s].add(r.split_group_id)
        rows_by_src[s][r.source_name or ""] += 1
        qtype_per[s][r.question_type] += 1
    for g, s in sorted(group_split.items()):
        groups_by_dom[s][dominant[g]] += 1
    group_sizes = Counter(r.split_group_id for r in records)
    files[f"{man}/split_manifest.meta.json"] = json_bytes(
        {
            "split_version": split_version,
            "corpus_version": corpus_version,
            "seed": cfg.split.seed,
            "ratios": cfg.split.ratios,
            "unit": "split_group_id",
            "stratified_by": "dominant source of the group (ties -> alphabetical)",
            "records": len(records),
            "split_groups": len(group_split),
            "duplicate_groups": len({r.duplicate_group_id for r in records}),
            "largest_split_group_records": max(group_sizes.values()),
            "rows_per_split": {s: rows_per.get(s, 0) for s in SPLITS},
            "groups_per_split": {s: len(groups_per.get(s, ())) for s in SPLITS},
            "rows_per_split_by_source": {s: dict(sorted(rows_by_src[s].items())) for s in SPLITS},
            "groups_per_split_by_dominant_source": {s: dict(sorted(groups_by_dom[s].items())) for s in SPLITS},
            "question_types_per_split": {s: dict(sorted(qtype_per[s].items())) for s in SPLITS},
            "question_type_rule_version": qrules.rule_version,
            "question_type_rules": qrules.table(),
            "question_type_default": qrules.default_type,
            "grouping": {
                **cfg.grouping.model_dump(),
                "boilerplate_answers": grouping.boilerplate_answers,
                "boilerplate_records": sum(grouping.boilerplate),
                "edges": grouping.edge_counts,
                "near_dup_pairs": len(grouping.near_dup_pairs),
                "template_near_dup_pairs": len(grouping.template_pairs),
                "residual_pairs": len(grouping.residual_pairs),
            },
            "grouping_disclosure": GROUPING_DISCLOSURE,
        }
    )

    leak = leakage_report(
        record_ids=rids,
        questions=questions,
        answers=answers,
        topics=topics,
        boilerplate=grouping.boilerplate,
        split_group_ids=[r.split_group_id for r in records],
        duplicate_group_ids=[r.duplicate_group_id for r in records],
        splits=splits,
        near_dup_pairs=grouping.near_dup_pairs,
        template_pairs=grouping.template_pairs,
        residual_pairs=grouping.residual_pairs,
    )
    leak = {"split_version": split_version, "corpus_version": corpus_version, **leak}
    files[f"{man}/leakage_report.json"] = json_bytes(leak)

    # ---- exports (full records incl. text; gitignored under data/processed)
    export_entries = {}
    for s in SPLITS:
        rel = f"{proc}/exports/records_{s}.jsonl"
        data = jsonl_bytes(r.model_dump(mode="json") for r, sp in zip(records, splits, strict=True) if sp == s)
        files[rel] = data
        export_entries[s] = {"path": rel, "records": rows_per.get(s, 0), "sha256": sha256_bytes(data),
                             "size_bytes": len(data)}
    files[f"{man}/exports_manifest.json"] = json_bytes(
        {
            "corpus_version": corpus_version,
            "split_version": split_version,
            "format": "JSONL, one medquad_qa.contracts.MedicalRecord per line (model_dump mode=json, sorted keys)",
            "contracts_version": CONTRACTS_VERSION,
            "splits": export_entries,
        }
    )

    dup_sizes = Counter(r.duplicate_group_id for r in records)
    dup_topics: dict[str, set[str | None]] = defaultdict(set)
    for r in records:
        dup_topics[r.duplicate_group_id].add(topic_key(r.topic))
    multi_topic_dups = [g for g, ts in dup_topics.items() if len(ts) > 1]
    files[f"{man}/audit.json"] = json_bytes(
        {
            "corpus_version": corpus_version,
            "split_version": split_version,
            "source_sha256": src_sha,
            "raw": audit_mod.raw_audit(rows),
            "exclusions": {
                "total": len(excluded),
                "by_reason": {k: excl_counts.get(k, 0) for k in EXCLUSION_REASONS},
                "by_detail": dict(sorted(excl_detail.items())),
                "by_reason_and_source": {
                    k: dict(sorted(Counter(e.source for e in excluded if e.reason == k).items()))
                    for k in EXCLUSION_REASONS
                },
            },
            "reconciliation": {"source_rows": len(rows), "corpus_records": len(records), "excluded": len(excluded)},
            "corpus": audit_mod.corpus_audit(records),
            "grouping": {
                "boilerplate_answers": grouping.boilerplate_answers,
                "boilerplate_records": sum(grouping.boilerplate),
                "duplicate_groups": len({r.duplicate_group_id for r in records}),
                "duplicate_groups_multi_record": sum(1 for n in dup_sizes.values() if n > 1),
                "duplicate_groups_multi_topic": len(multi_topic_dups),
                "records_in_multi_topic_duplicate_groups": sum(dup_sizes[g] for g in multi_topic_dups),
                "split_groups": len(group_split),
                "split_group_size": audit_mod.distribution(list(group_sizes.values())),
                "near_dup_pairs_j_ge_threshold": len(grouping.near_dup_pairs),
                "template_near_dup_pairs_j_ge_threshold": len(grouping.template_pairs),
                "residual_template_pairs_j_0_5_to_threshold": len(grouping.residual_pairs),
                "edges": grouping.edge_counts,
            },
        }
    )

    file_groups = {
        "build": sorted(files),
        "audit": [f"{man}/audit.json", f"{man}/source_dataset.json"],
        "split": [f"{man}/split_manifest.jsonl", f"{man}/split_manifest.meta.json", f"{man}/leakage_report.json"],
        "export": [f"{proc}/exports/records_{s}.jsonl" for s in SPLITS] + [f"{man}/exports_manifest.json"],
    }
    return BuildResult(records, splits, excluded, corpus_version, split_version, leak, files, file_groups)
