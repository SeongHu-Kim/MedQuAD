"""Duplicate groups and leakage-control split groups (union-find over record indices).

duplicate_group: records with the same non-boilerplate normalized answer, the same (question, answer)
  pair, or a near-duplicate answer (plain word-shingle Jaccard >= threshold). Used for retrieval
  collapse, so answers about *different* conditions must not merge here.
split_group: every duplicate-group edge, plus the same folded topic key across sources (also after removing
  parenthetical aliases, so "X" and "X (abbrev)" link), the same
  normalized question key, and template near-duplicates (Jaccard >= threshold after masking each
  record's own topic name, e.g. the same inheritance template written for two conditions). So each
  duplicate group lies inside one split group, and split groups are deliberately conservative.
Boilerplate answers (shared by >= ``boilerplate_min_topics`` topics that do not all start with the same
word) link neither: one generic sentence would otherwise chain unrelated topics together. They carry the
``boilerplate_answer`` flag instead. An answer shared only within one topic family ("X type 1", "X type 2")
is content and does link.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

from medquad_qa.data.config import GroupingConfig
from medquad_qa.data.ids import group_id
from medquad_qa.data.neardup import masked_shingles, similar_pairs
from medquad_qa.data.normalize import bracket_stripped_topic_key, match_key, topic_key


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def union_all(self, members: Sequence[int]) -> None:
        for m in members[1:]:
            self.union(members[0], m)

    def components(self) -> dict[int, list[int]]:
        comps: dict[int, list[int]] = defaultdict(list)
        for i in range(len(self.parent)):
            comps[self.find(i)].append(i)
        return comps


@dataclass(frozen=True)
class GroupInput:
    record_id: str
    question: str  # normalized
    answer: str  # normalized
    topic: str | None


@dataclass(frozen=True)
class RecordPair:
    a: int
    b: int
    jaccard: float


@dataclass
class GroupingResult:
    duplicate_group_ids: list[str]
    split_group_ids: list[str]
    boilerplate: list[bool]
    boilerplate_answers: int
    near_dup_pairs: list[RecordPair]  # plain shingles, J >= near_dup_jaccard, non-boilerplate
    template_pairs: list[RecordPair]  # topic-masked shingles, J >= near_dup_jaccard, non-boilerplate
    residual_pairs: list[RecordPair]  # topic-masked, residual_jaccard_min <= J < near_dup_jaccard
    edge_counts: dict[str, int]


def _buckets(keys: Sequence[Hashable | None]) -> list[list[int]]:
    idx: dict[Hashable, list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        if k is not None:
            idx[k].append(i)
    return [v for v in idx.values() if len(v) > 1]


def _one_family(topic_keys: set[str | None]) -> bool:
    """True when all topics start with the same word, e.g. "noonan syndrome 1" / "noonan syndrome 2".

    An answer shared only within such a family is disease-specific content (subtypes copied from one
    parent page), so it must link groups rather than be treated as boilerplate.
    """
    if None in topic_keys:
        return False
    first_words = {tk.split(" ", 1)[0] for tk in topic_keys if tk}
    return len(first_words) == 1


def _assign_ids(uf: UnionFind, record_ids: Sequence[str], prefix: str) -> list[str]:
    out = [""] * len(record_ids)
    for members in uf.components().values():
        gid = group_id(prefix, (record_ids[m] for m in members))
        for m in members:
            out[m] = gid
    return out


def _similar_records(
    items: Sequence[GroupInput], eligible: Sequence[bool], cfg: GroupingConfig, *, mask_topic: bool
) -> tuple[list[RecordPair], list[RecordPair]]:
    """Return (pairs with J >= near_dup_jaccard, pairs with residual_jaccard_min <= J < near_dup_jaccard).

    Records with identical shingle sets are collapsed into one unit before pair search and linked
    to each other with J = 1.0.
    """
    unit_of: dict[frozenset[str], int] = {}
    unit_sets: list[frozenset[str]] = []
    unit_members: list[list[int]] = []
    for i, it in enumerate(items):
        if not eligible[i]:
            continue
        s = masked_shingles(it.answer, it.topic if mask_topic else None, cfg.shingle_size)
        if not s:
            continue
        u = unit_of.setdefault(s, len(unit_sets))
        if u == len(unit_sets):
            unit_sets.append(s)
            unit_members.append([])
        unit_members[u].append(i)
    pairs = similar_pairs(
        unit_sets,
        max_df=cfg.rare_shingle_max_df,
        min_shared_rare=cfg.min_shared_rare_shingles,
        min_jaccard=cfg.residual_jaccard_min,
    )
    near: list[RecordPair] = []
    residual: list[RecordPair] = []
    for p in pairs:
        a, b = sorted((unit_members[p.i][0], unit_members[p.j][0]))
        (near if p.jaccard >= cfg.near_dup_jaccard else residual).append(RecordPair(a, b, p.jaccard))
    for members in unit_members:
        near.extend(RecordPair(members[0], m, 1.0) for m in members[1:])
    near.sort(key=lambda r: (r.a, r.b))
    residual.sort(key=lambda r: (r.a, r.b))
    return near, residual


def build_groups(items: Sequence[GroupInput], cfg: GroupingConfig) -> GroupingResult:
    n = len(items)
    record_ids = [it.record_id for it in items]
    tkeys = [topic_key(it.topic) for it in items]

    topics_per_answer: dict[str, set[str | None]] = defaultdict(set)
    for it, tk in zip(items, tkeys, strict=True):
        topics_per_answer[it.answer].add(tk)
    boiler_answers = {
        a for a, ts in topics_per_answer.items() if len(ts) >= cfg.boilerplate_min_topics and not _one_family(ts)
    }
    boilerplate = [it.answer in boiler_answers for it in items]
    eligible = [not b for b in boilerplate]

    dup = UnionFind(n)
    split = UnionFind(n)
    edges = {
        "exact_answer": 0,
        "exact_qa_pair": 0,
        "near_dup": 0,
        "topic": 0,
        "bracket_topic": 0,
        "question": 0,
        "template_near_dup": 0,
    }

    for members in _buckets([it.answer if ok else None for it, ok in zip(items, eligible, strict=True)]):
        dup.union_all(members)
        split.union_all(members)
        edges["exact_answer"] += len(members) - 1
    for members in _buckets([(it.question, it.answer) for it in items]):
        dup.union_all(members)
        split.union_all(members)
        edges["exact_qa_pair"] += len(members) - 1
    near, _ = _similar_records(items, eligible, cfg, mask_topic=False)
    for rp in near:
        dup.union(rp.a, rp.b)
        split.union(rp.a, rp.b)
    edges["near_dup"] = len(near)

    for members in _buckets(tkeys):
        split.union_all(members)
        edges["topic"] += len(members) - 1
    # "X" and "X (alias)" name the same condition (evaluator CR3): link on the bracket-stripped key too.
    for members in _buckets([bracket_stripped_topic_key(it.topic) for it in items]):
        split.union_all(members)
        edges["bracket_topic"] += len(members) - 1
    for members in _buckets([match_key(it.question) for it in items]):
        split.union_all(members)
        edges["question"] += len(members) - 1
    template, residual = _similar_records(items, eligible, cfg, mask_topic=True)
    for rp in template:
        split.union(rp.a, rp.b)
    edges["template_near_dup"] = len(template)

    return GroupingResult(
        duplicate_group_ids=_assign_ids(dup, record_ids, "dg"),
        split_group_ids=_assign_ids(split, record_ids, "sg"),
        boilerplate=boilerplate,
        boilerplate_answers=len(boiler_answers),
        near_dup_pairs=near,
        template_pairs=template,
        residual_pairs=residual,
        edge_counts=edges,
    )
