"""Cross-split leakage checks. Any non-zero blocking check fails the build/verify."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence
from typing import Any

from medquad_qa.data.grouping import RecordPair
from medquad_qa.data.normalize import bracket_stripped_topic_key, match_key, topic_key


def _cross_split_keys(keys: Sequence[Hashable | None], splits: Sequence[str]) -> list[Hashable]:
    seen: dict[Hashable, set[str]] = defaultdict(set)
    for k, s in zip(keys, splits, strict=True):
        if k is not None:
            seen[k].add(s)
    return [k for k, ss in seen.items() if len(ss) > 1]


def leakage_report(
    *,
    record_ids: Sequence[str],
    questions: Sequence[str],
    answers: Sequence[str],
    topics: Sequence[str | None],
    boilerplate: Sequence[bool],
    split_group_ids: Sequence[str],
    duplicate_group_ids: Sequence[str],
    splits: Sequence[str],
    near_dup_pairs: Sequence[RecordPair],
    template_pairs: Sequence[RecordPair],
    residual_pairs: Sequence[RecordPair],
) -> dict[str, Any]:
    """Blocking checks must all be 0. Diagnostics are reported, never silently fixed."""
    non_boiler_answers = [a if not b else None for a, b in zip(answers, boilerplate, strict=True)]
    blocking = {
        "split_group_cross_split": len(_cross_split_keys(split_group_ids, splits)),
        "duplicate_group_cross_split": len(_cross_split_keys(duplicate_group_ids, splits)),
        "topic_key_cross_split": len(_cross_split_keys([topic_key(t) for t in topics], splits)),
        "normalized_question_cross_split": len(_cross_split_keys([match_key(q) for q in questions], splits)),
        "non_boilerplate_answer_cross_split": len(_cross_split_keys(non_boiler_answers, splits)),
        "near_dup_pairs_cross_split": sum(1 for p in near_dup_pairs if splits[p.a] != splits[p.b]),
        "template_near_dup_pairs_cross_split": sum(1 for p in template_pairs if splits[p.a] != splits[p.b]),
    }
    boiler_cross = _cross_split_keys([a if b else None for a, b in zip(answers, boilerplate, strict=True)], splits)
    residual_cross = [p for p in residual_pairs if splits[p.a] != splits[p.b]]

    # Diagnostic: topics that only merge after removing parenthetical text, and straddle splits.
    bracket_topics: dict[str, set[str]] = defaultdict(set)
    for t in topics:
        bk = bracket_stripped_topic_key(t)
        tk = topic_key(t)
        if bk is not None and tk is not None:
            bracket_topics[bk].add(tk)
    split_of_topic: dict[str, set[str]] = defaultdict(set)
    for t, s in zip(topics, splits, strict=True):
        tk = topic_key(t)
        if tk is not None:
            split_of_topic[tk].add(s)
    bracket_merges = []
    for bk, tks in sorted(bracket_topics.items()):
        if len(tks) > 1:
            ss = sorted(set().union(*(split_of_topic[tk] for tk in tks)))
            bracket_merges.append({"bracket_stripped_key": bk, "topic_keys": sorted(tks), "splits": ss})

    return {
        "blocking_checks": blocking,
        "passed": all(v == 0 for v in blocking.values()),
        "diagnostics": {
            "boilerplate_answers_cross_split": len(boiler_cross),
            "residual_pairs_total": len(residual_pairs),
            "residual_pairs_cross_split": len(residual_cross),
            "residual_pairs_cross_split_list": [
                {
                    "record_id_a": record_ids[p.a],
                    "split_a": splits[p.a],
                    "record_id_b": record_ids[p.b],
                    "split_b": splits[p.b],
                    "jaccard": round(p.jaccard, 4),
                }
                for p in residual_cross
            ],
            "bracket_stripped_topic_merges": len(bracket_merges),
            "bracket_stripped_topic_merges_cross_split": sum(1 for m in bracket_merges if len(m["splits"]) > 1),
            "bracket_stripped_topic_merge_list": bracket_merges,
        },
    }
