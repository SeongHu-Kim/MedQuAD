"""Near-dup, grouping, split and leakage unit tests (SYNTHETIC inputs only)."""

import pytest

from medquad_qa.data.config import GroupingConfig
from medquad_qa.data.grouping import GroupInput, UnionFind, build_groups
from medquad_qa.data.leakage import leakage_report
from medquad_qa.data.neardup import jaccard, mask_topic, masked_shingles, similar_pairs, tokens
from medquad_qa.data.splits import SPLITS, assign_splits, dominant_source

TEMPLATE = (
    "{t} is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell "
    "carry a SYNTHETIC change. The parents each carry one copy but usually show no signs of it."
)
# The name recurs, so plain-shingle Jaccard between two conditions is < 0.8 while the masked one is 1.0.
NAMED = "{t} is SYNTHETIC. People with {t} may tire. {t} is not inherited. Doctors treat {t} with rest and fluids."
BOILER = "SYNTHETIC boilerplate text about where to find more resources on this topic online today."


def test_mask_and_jaccard() -> None:
    assert mask_topic(tokens("Foo bar is rare. foo bar"), tokens("Foo Bar")) == ["<t>", "is", "rare", "<t>"]
    a = masked_shingles(TEMPLATE.format(t="Gammaemia"), "Gammaemia", 5)
    b = masked_shingles(TEMPLATE.format(t="Deltapathy"), "Deltapathy", 5)
    assert a == b and jaccard(a, b) == 1.0
    assert jaccard(frozenset(), a) == 0.0
    assert masked_shingles("too short", None, 5) == frozenset()


def test_similar_pairs_threshold() -> None:
    s1 = frozenset(f"s{i}" for i in range(20))
    s2 = frozenset(f"s{i}" for i in range(2, 22))  # J = 18/22
    s3 = frozenset(f"x{i}" for i in range(20))
    pairs = similar_pairs([s1, s2, s3], max_df=10, min_shared_rare=5, min_jaccard=0.8)
    assert [(p.i, p.j) for p in pairs] == [(0, 1)]
    assert similar_pairs([s1, s2], max_df=1, min_shared_rare=5, min_jaccard=0.0) == []  # nothing rare


def test_union_find() -> None:
    uf = UnionFind(5)
    uf.union(3, 4)
    uf.union_all([0, 2, 4])
    comps = sorted(sorted(v) for v in uf.components().values())
    assert comps == [[0, 2, 3, 4], [1]]


def _items() -> list[GroupInput]:
    return [
        GroupInput("mq-0000000000000000", "What is A?", "A text one about alpha only here now ok.", "Alpha"),
        GroupInput("mq-0000000000000001", "What is A?", "A text one about alpha only here now ok.", "ALPHA "),
        GroupInput("mq-0000000000000002", "Is G inherited?", TEMPLATE.format(t="Gammaemia"), "Gammaemia"),
        GroupInput("mq-0000000000000003", "Is D inherited?", TEMPLATE.format(t="Deltapathy"), "Deltapathy"),
        GroupInput("mq-0000000000000004", "What is E?", BOILER, "Eta"),
        GroupInput("mq-0000000000000005", "What is T?", BOILER, "Theta"),
        GroupInput("mq-0000000000000006", "What is I?", BOILER, "Iota"),
        GroupInput("mq-0000000000000007", "What causes Alpha?", "Different SYNTHETIC cause text.", "alpha"),
        GroupInput("mq-0000000000000008", "What is K?", NAMED.format(t="Kappaitis"), "Kappaitis"),
        GroupInput("mq-0000000000000009", "What is L?", NAMED.format(t="Lambdosis"), "Lambdosis"),
    ]


def test_build_groups_semantics() -> None:
    g = build_groups(_items(), GroupingConfig())
    d, s = g.duplicate_group_ids, g.split_group_ids
    assert d[0] == d[1] and s[0] == s[1] == s[7]  # same answer; topic folded across case
    assert d[7] != d[0]  # same topic is not a duplicate
    assert d[2] == d[3] and s[2] == s[3]  # one name in a long shared text: plain near-dup (J >= 0.8)
    assert d[8] != d[9] and s[8] == s[9]  # template near-dup only: split group, not duplicate group
    assert g.boilerplate == [False, False, False, False, True, True, True, False, False, False]
    assert len({s[4], s[5], s[6]}) == 3 and len({d[4], d[5], d[6]}) == 3  # boilerplate links nothing
    # every duplicate group lies inside one split group
    for i in range(len(d)):
        for j in range(len(d)):
            if d[i] == d[j]:
                assert s[i] == s[j]
    assert all(x.startswith("dg-") for x in d) and all(x.startswith("sg-") for x in s)


def test_dominant_source_tie_break() -> None:
    assert dominant_source(["B", "A", "B", "A"]) == "A"
    assert dominant_source(["C", "B", "C"]) == "C"


def test_assign_splits_deterministic_and_group_level() -> None:
    groups = [f"sg-{i // 3:03d}" for i in range(300)]
    sources = ["S1" if i % 2 else "S2" for i in range(300)]
    ratios = {"train": 0.8, "validation": 0.1, "test": 0.1}
    a1, dom = assign_splits(groups, sources, ratios, seed=7)
    a2, _ = assign_splits(groups, sources, ratios, seed=7)
    a3, _ = assign_splits(groups, sources, ratios, seed=8)
    assert a1 == a2 and a1 != a3
    assert set(a1) == set(groups) and set(dom) == set(groups)
    counts = {s: sum(1 for g in groups if a1[g] == s) for s in SPLITS}
    assert 225 <= counts["train"] <= 255 and counts["validation"] >= 21 and counts["test"] >= 21


def test_assign_splits_rejects_bad_ratios() -> None:
    with pytest.raises(ValueError):
        assign_splits(["g"], ["s"], {"train": 0.9, "test": 0.1}, seed=1)


def test_leakage_report_detects_overlap() -> None:
    items = _items()
    g = build_groups(items, GroupingConfig())
    common = dict(
        record_ids=[i.record_id for i in items],
        questions=[i.question for i in items],
        answers=[i.answer for i in items],
        topics=[i.topic for i in items],
        boilerplate=g.boilerplate,
        near_dup_pairs=g.near_dup_pairs,
        template_pairs=g.template_pairs,
        residual_pairs=g.residual_pairs,
    )
    by_group = {sg: SPLITS[k % 3] for k, sg in enumerate(sorted(set(g.split_group_ids)))}
    good = [by_group[sg] for sg in g.split_group_ids]
    rep = leakage_report(
        split_group_ids=g.split_group_ids, duplicate_group_ids=g.duplicate_group_ids, splits=good, **common
    )
    assert rep["passed"], rep["blocking_checks"]

    bad = list(good)
    bad[1] = next(s for s in SPLITS if s != good[0])  # split the exact-duplicate pair
    rep = leakage_report(
        split_group_ids=g.split_group_ids, duplicate_group_ids=g.duplicate_group_ids, splits=bad, **common
    )
    assert not rep["passed"]
    b = rep["blocking_checks"]
    assert b["non_boilerplate_answer_cross_split"] == 1 and b["topic_key_cross_split"] == 1
    assert b["normalized_question_cross_split"] == 1 and b["split_group_cross_split"] == 1


def test_family_shared_answer_is_not_boilerplate() -> None:
    """F-001 regression: one answer copied across numbered subtypes is content and must link them."""
    fam = "Noonix syndrome is a SYNTHETIC disorder affecting many parts of the body in several ways."
    items = [
        GroupInput(f"mq-00000000000000a{i}", f"What is Noonix syndrome {i}?", fam, f"Noonix syndrome {i}")
        for i in range(1, 5)
    ]
    g = build_groups(items, GroupingConfig())
    assert g.boilerplate == [False] * 4
    assert len(set(g.split_group_ids)) == 1 and len(set(g.duplicate_group_ids)) == 1


_CONTENT = (
    "SYNTHETIC prevention trial text: researchers test whether a daily pill lowers the chance of a new tumour "
    "in people at high risk, and volunteers are followed for ten years with yearly scans and blood tests."
)
# One pair per F-001 class (+ CR3 bracket alias). Each pair must be linked AND, if split apart, detected.
F001_CASES = {
    "topic_key_cross_split": (
        GroupInput("mq-00000000000000b1", "What causes it?", "SYNTHETIC answer one.", "Coffin-Lowry syndrome"),
        GroupInput("mq-00000000000000b2", "How is it treated?", "SYNTHETIC answer two.", "Coffin Lowry Syndrome"),
    ),
    "normalized_question_cross_split": (
        GroupInput("mq-00000000000000c1", "What is Graves' disease?", "SYNTHETIC answer three.", None),
        GroupInput("mq-00000000000000c2", "What is Graves disease ?", "SYNTHETIC answer four.", None),
    ),
    "non_boilerplate_answer_cross_split": (  # family-shared answer (3 subtypes) is content, not boilerplate
        GroupInput("mq-00000000000000d1", "What is N1?", "SYNTHETIC Noonix text shared.", "Noonix syndrome 1"),
        GroupInput("mq-00000000000000d2", "What is N2?", "SYNTHETIC Noonix text shared.", "Noonix syndrome 2"),
        GroupInput("mq-00000000000000d3", "What is N3?", "SYNTHETIC Noonix text shared.", "Noonix syndrome 3"),
    ),
    "near_dup_pairs_cross_split": (  # same content, different topics, not exact
        GroupInput("mq-00000000000000e1", "Prevention of A?", _CONTENT, "Endometrial cancer"),
        GroupInput(
            "mq-00000000000000e2", "Prevention of O?", _CONTENT + " Results are expected soon.", "Ovarian cancer"
        ),
    ),
    "bracket_stripped_topic_cross_split": (
        GroupInput("mq-00000000000000f1", "What is CFS?", "SYNTHETIC answer five.", "Chronic fatigue syndrome"),
        GroupInput("mq-00000000000000f2", "Who gets CFS?", "SYNTHETIC answer six.", "Chronic Fatigue Syndrome (CFS)"),
    ),
}


@pytest.mark.parametrize("check", sorted(F001_CASES))
def test_f001_classes_linked_and_detected(check: str) -> None:
    items = list(F001_CASES[check])
    g = build_groups(items, GroupingConfig())
    assert len(set(g.split_group_ids)) == 1, "grouping must keep the pair in one split group"
    assert not any(g.boilerplate)
    splits = ["train"] + ["test"] * (len(items) - 1)  # force the leak
    rep = leakage_report(
        record_ids=[i.record_id for i in items],
        questions=[i.question for i in items],
        answers=[i.answer for i in items],
        topics=[i.topic for i in items],
        boilerplate=g.boilerplate,
        split_group_ids=g.split_group_ids,
        duplicate_group_ids=g.duplicate_group_ids,
        splits=splits,
        near_dup_pairs=g.near_dup_pairs,
        template_pairs=g.template_pairs,
        residual_pairs=g.residual_pairs,
    )
    assert not rep["passed"]
    assert rep["blocking_checks"][check] >= 1
