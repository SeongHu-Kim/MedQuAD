"""Dataset quality audit. Produces counts and distributions only (no dataset text)."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

from medquad_qa.contracts import MedicalRecord
from medquad_qa.data.ingest import RawRow
from medquad_qa.data.normalize import match_key, normalize_question, normalize_text, word_count

PERCENTILES = (0, 5, 25, 50, 75, 95, 100)


def distribution(values: Sequence[int]) -> dict[str, float | int]:
    """Nearest-rank percentiles plus mean; empty input gives count 0."""
    if not values:
        return {"count": 0}
    v = sorted(values)
    out: dict[str, float | int] = {"count": len(v), "mean": round(sum(v) / len(v), 2)}
    for p in PERCENTILES:
        idx = min(len(v) - 1, max(0, round(p / 100 * (len(v) - 1))))
        out[f"p{p}"] = v[idx]
    return out


def _extra_copies(keys: Sequence[object]) -> dict[str, int]:
    c = Counter(keys)
    dups = {k: n for k, n in c.items() if n > 1}
    return {
        "distinct_values_repeated": len(dups),
        "rows_in_repeated_values": sum(dups.values()),
        "extra_copies": sum(n - 1 for n in dups.values()),
    }


def raw_audit(rows: Sequence[RawRow]) -> dict[str, Any]:
    """Audit of the raw file before any exclusion."""
    by_source = Counter(r.source for r in rows)
    nq = [normalize_question(r.question) for r in rows]
    na = [normalize_text(r.answer) for r in rows]
    nonempty = [i for i, a in enumerate(na) if a]
    per_source_words: dict[str, list[int]] = defaultdict(list)
    for i in nonempty:
        per_source_words[rows[i].source].append(word_count(na[i]))
    return {
        "rows": len(rows),
        "rows_by_source": dict(sorted(by_source.items())),
        "missing": {
            "empty_question": sum(1 for q in nq if not q),
            "empty_answer": sum(1 for a in na if not a),
            "empty_focus_area": sum(1 for r in rows if not r.focus_area.strip()),
            "empty_source": sum(1 for r in rows if not r.source.strip()),
        },
        "normalization_changes": {
            "question_changed": sum(1 for r, q in zip(rows, nq, strict=True) if r.question != q),
            "answer_changed": sum(1 for r, a in zip(rows, na, strict=True) if r.answer != a),
        },
        "exact_duplicates": {
            "full_row": _extra_copies([(r.source, r.focus_area, r.question, r.answer) for r in rows]),
            "normalized_question_answer_pair": _extra_copies([(q, a) for q, a in zip(nq, na, strict=True) if a]),
            "normalized_question_key": _extra_copies([match_key(q) for q in nq]),
            "normalized_answer": _extra_copies([na[i] for i in nonempty]),
        },
        "lengths_raw_nonempty": {
            "question_words": distribution([word_count(q) for q in nq]),
            "answer_words": distribution([word_count(na[i]) for i in nonempty]),
            "answer_chars": distribution([len(na[i]) for i in nonempty]),
            "answer_words_by_source": {s: distribution(v) for s, v in sorted(per_source_words.items())},
        },
    }


def corpus_audit(records: Sequence[MedicalRecord]) -> dict[str, Any]:
    flags: Counter[str] = Counter()
    flags_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    for r in records:
        flags.update(r.quality_flags)
        flags_by_source[r.source_name or "unknown"].update(r.quality_flags)
    qtypes = Counter(r.question_type for r in records)
    qtype_by_source: dict[str, Counter[str | None]] = defaultdict(Counter)
    for r in records:
        qtype_by_source[r.source_name or "unknown"][r.question_type] += 1
    return {
        "records": len(records),
        "records_by_source": dict(sorted(Counter(r.source_name for r in records).items())),
        "distinct_topics": len({r.topic for r in records if r.topic}),
        "quality_flags": dict(sorted(flags.items())),
        "quality_flags_by_source": {s: dict(sorted(c.items())) for s, c in sorted(flags_by_source.items())},
        "question_types": dict(sorted(qtypes.items())),
        "question_types_by_source": {s: dict(sorted(c.items())) for s, c in sorted(qtype_by_source.items())},
        "lengths": {
            "question_words": distribution([word_count(r.question) for r in records]),
            "answer_words": distribution([word_count(r.answer) for r in records]),
            "answer_chars": distribution([len(r.answer) for r in records]),
        },
    }
