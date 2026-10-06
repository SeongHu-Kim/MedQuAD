"""Build the frozen evaluation sets (E2) and the 50-item non-clinical spot-check sheet.

Usage:
  python scripts/evaluation/build_evalsets.py [--out artifacts/evaluation/evalsets] [--check-only]

Writes <out>/{test_track_a,train_probe,test_track_c,dev}.jsonl (+ .manifest.json each), <out>/build_report.json,
and artifacts/evaluation/spotcheck/per_item/spotcheck_sheet.csv (gitignored: it shows dataset questions).
Fails (exit 1) on any check_leakage error. Deterministic for a given split_version and agent_items.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from medquad_qa.contracts.evaluation import EvaluationExample
from medquad_qa.evaluation.evalset import sha256_file, write_eval_set
from medquad_qa.evaluation.evalset_builder import (
    BUILDER_VERSION,
    HARD_NEGATIVE_QTYPES,
    SEED,
    TEMPLATE_RULE_VERSION,
    TEMPLATES,
    Corpus,
    Rec,
    assert_absent,
    eligible,
    fold,
    pick_one_per_group,
    stable_id,
    template_question,
    track_a_example,
)
from medquad_qa.evaluation.leakage import check_leakage

ROOT = Path(__file__).resolve().parents[2]


def _rng(offset: int) -> random.Random:
    """Seeded, reproducible sampling (not used for anything security-relevant)."""
    return random.Random(SEED + offset)  # noqa: S311


AGENT = ROOT / "configs/evaluation/agent_items.json"


def ex(**kw: Any) -> EvaluationExample:
    base: dict[str, Any] = {
        "label_provenance": "synthetic_rule",
        "reviewer": "none",
        "evaluation_track": "answerability_robustness",
    }
    base.update(kw)
    return EvaluationExample(**base)


def track_a(
    corpus: Corpus, recs: list[Rec], agent_q: dict[str, str], split: str, prefix: str
) -> list[EvaluationExample]:
    rng = _rng(11)
    out = []
    for i, r in enumerate(recs):
        if i % 3 == 2:
            out.append(
                track_a_example(
                    r, corpus, agent_q[r.record_id], agent=True, eval_split=split, method="agent_written", prefix=prefix
                )
            )
        else:
            q, t = template_question(r, rng)
            out.append(
                track_a_example(
                    r, corpus, q, agent=False, eval_split=split, method=f"{TEMPLATE_RULE_VERSION}:{t}", prefix=prefix
                )
            )
    return out


def topic_recs(corpus: Corpus, split: str, exclude_groups: set[str]) -> dict[str, Rec]:
    """One representative eligible record per folded topic in ``split`` (deterministic)."""
    reps: dict[str, Rec] = {}
    for r in sorted(corpus.records, key=lambda x: x.record_id):
        if r.split == split and eligible(r) and r.split_group_id not in exclude_groups:
            reps.setdefault(fold(r.topic), r)
    return reps


def personal_and_controls(
    corpus: Corpus, templates: list[str], split: str, prefix: str, rng: random.Random, used_topics: set[str]
) -> list[EvaluationExample]:
    eval_split = "test" if split == "test" else "dev"
    reps = topic_recs(corpus, split, set())
    keys = sorted(k for k in reps if k not in used_topics)
    rng.shuffle(keys)
    out = []
    for tmpl, key in zip(templates, keys, strict=False):
        used_topics.add(key)
        rep = reps[key]
        pid = stable_id(f"{prefix}-pa", rep.record_id, tmpl)
        out.append(
            ex(
                example_id=pid,
                question=tmpl.format(t=rep.topic),
                answerable=False,
                expected_behavior="abstain",
                expected_abstention_reason="personalized_medical_advice",
                question_provenance="llm_generated_unreviewed",
                paraphrase_method="agent_written_template",
                derived_from_record_ids=[rep.record_id],
                split_group_id=rep.split_group_id,
                eval_split=eval_split,
                case_type="personalized_advice",
                topic=rep.topic,
                notes="personalized_advice:first_person_decision",
            )
        )
        # matched general-information control on the same topic (rule-generated, answerable from the corpus)
        qtypes = sorted(q for q in corpus.qtypes_by_topic[key] if q in TEMPLATES)
        qt = "treatment" if "treatment" in qtypes else ("information" if "information" in qtypes else qtypes[0])
        pool = sorted(corpus.by_topic_qtype[(key, qt)], key=lambda r: (not eligible(r), r.record_id))
        ctrl_rec = pool[0]
        q, t = template_question(ctrl_rec, rng)
        out.append(
            ex(
                example_id=stable_id(f"{prefix}-gc", ctrl_rec.record_id, q),
                question=q,
                gold_record_ids=corpus.gold(ctrl_rec),
                answerable=True,
                expected_behavior="answer",
                question_provenance="synthetic_rule",
                paraphrase_method=f"{TEMPLATE_RULE_VERSION}:{t}",
                derived_from_record_ids=[ctrl_rec.record_id],
                split_group_id=ctrl_rec.split_group_id,
                eval_split=eval_split,
                case_type="answerable",
                question_type=qt,
                topic=ctrl_rec.topic,
                notes=f"general_control:pair={pid}",
            )
        )
    if len(out) != 2 * len(templates):
        raise ValueError("not enough topics for personal/control pairs")
    return out


def hard_negatives(
    corpus: Corpus, n: int, split: str, prefix: str, rng: random.Random, used_topics: set[str]
) -> list[EvaluationExample]:
    eval_split = "test" if split == "test" else "dev"
    reps = topic_recs(corpus, split, set())
    keys = sorted(k for k in reps if k not in used_topics)
    rng.shuffle(keys)
    out = []
    for key in keys:
        missing = [q for q in HARD_NEGATIVE_QTYPES if q not in corpus.qtypes_by_topic[key]]
        if not missing:
            continue
        qt = missing[rng.randrange(len(missing))]
        rep = reps[key]
        q = TEMPLATES[qt][rng.randrange(len(TEMPLATES[qt]))].format(t=rep.topic)
        used_topics.add(key)
        out.append(
            ex(
                example_id=stable_id(f"{prefix}-hn", rep.record_id, qt),
                question=q,
                answerable=False,
                expected_behavior="abstain",
                expected_abstention_reason="insufficient_evidence",
                question_provenance="synthetic_rule",
                paraphrase_method=f"{TEMPLATE_RULE_VERSION}:hard_negative",
                derived_from_record_ids=[rep.record_id],
                split_group_id=rep.split_group_id,
                eval_split=eval_split,
                case_type="unanswerable",
                question_type=qt,
                topic=rep.topic,
                notes=f"hard_negative:missing_qtype={qt}",
            )
        )
        if len(out) == n:
            return out
    raise ValueError(f"only {len(out)} hard negatives available in {split}")


def simple_items(
    questions: list[str], prefix: str, eval_split: str, *, case_type: str, note: str, behavior: str, reason: Any
) -> list[EvaluationExample]:
    return [
        ex(
            example_id=stable_id(prefix, q),
            question=q,
            answerable=False,
            expected_behavior=behavior,
            expected_abstention_reason=reason,
            question_provenance="llm_generated_unreviewed",
            paraphrase_method="agent_written",
            eval_split=eval_split,
            case_type=case_type,
            notes=note,
        )
        for q in questions
    ]


def fixture_items(agent: dict[str, Any]) -> list[EvaluationExample]:
    out = []
    for it in agent["conflicting_test"]:
        out.append(
            ex(
                example_id=stable_id("c-cf", it["q"]),
                question=it["q"],
                answerable=False,
                expected_behavior="either",
                question_provenance="llm_generated_unreviewed",
                paraphrase_method="agent_written",
                eval_split="test",
                case_type="conflicting_evidence",
                injected_evidence=it["ev"],
                notes="conflicting:fixture_retriever",
            )
        )
    for it in agent["adversarial_test"]:
        canary = "CANARY" + stable_id("k", it["q"])[2:10].upper()
        out.append(
            ex(
                example_id=stable_id("c-adv", it["q"]),
                question=it["q"],
                answerable=True,
                expected_behavior="answer",
                question_provenance="llm_generated_unreviewed",
                paraphrase_method="agent_written",
                eval_split="test",
                case_type="adversarial",
                injected_evidence=[f"{it['fact']} {it['inject'].format(c=canary)}"],
                notes=f"adversarial:canary={canary}",
            )
        )
    return out


def spotcheck_rows(sets: dict[str, list[EvaluationExample]], corpus: Corpus) -> list[dict[str, str]]:
    rng = _rng(99)
    a = sets["test_track_a"]
    c = sets["test_track_c"]
    picks: list[EvaluationExample] = []
    picks += rng.sample([e for e in a if e.question_provenance == "synthetic_rule"], 10)
    picks += rng.sample([e for e in a if e.question_provenance != "synthetic_rule"], 10)
    picks += rng.sample([e for e in c if e.case_type == "personalized_advice"], 5)
    picks += rng.sample([e for e in c if (e.notes or "").startswith("general_control")], 5)
    picks += rng.sample([e for e in c if (e.notes or "").startswith("hard_negative")], 10)
    picks += rng.sample([e for e in c if (e.notes or "").startswith("unanswerable")], 5)
    picks += rng.sample([e for e in c if e.case_type == "ambiguous"], 5)
    rows = []
    for e in picks:
        src = corpus.by_id.get(e.derived_from_record_ids[0]) if e.derived_from_record_ids else None
        rows.append(
            {
                "example_id": e.example_id,
                "case_type": e.case_type,
                "notes": e.notes or "",
                "eval_question": e.question,
                "expected_behavior": e.expected_behavior,
                "expected_reason": e.expected_abstention_reason or "",
                "topic": e.topic or "",
                "question_type": e.question_type or "",
                "source_dataset_question": src.question if src else "",
                "n_gold_records": str(len(e.gold_record_ids)),
                "reviewer_label_ok (yes/no/unsure)": "",
                "reviewer_comment": "",
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artifacts/evaluation/evalsets"))
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    corpus, meta = Corpus.load(ROOT)
    agent = json.loads(AGENT.read_text(encoding="utf-8"))

    test_recs = pick_one_per_group(corpus, "test", 300, _rng(0), set())
    dev_recs = pick_one_per_group(corpus, "validation", 30, _rng(1), set())
    probe_recs = pick_one_per_group(corpus, "train", 100, _rng(2), set())

    sets: dict[str, list[EvaluationExample]] = {}
    sets["test_track_a"] = track_a(corpus, test_recs, agent["track_a_paraphrases"], "test", "a")
    rng_p = _rng(3)
    sets["train_probe"] = []
    for r in probe_recs:
        q, t = template_question(r, rng_p)
        sets["train_probe"].append(
            track_a_example(
                r, corpus, q, agent=False, eval_split="train_probe", method=f"{TEMPLATE_RULE_VERSION}:{t}", prefix="p"
            )
        )

    # Track C (TEST) -------------------------------------------------------------------------------
    rng_c = _rng(21)
    used: set[str] = {fold(r.topic) for r in test_recs}  # Track C topics are disjoint from Track A topics
    corpus_text = " " + " ".join(fold(r.question) + " " + fold(r.topic) for r in corpus.records) + " "
    with (ROOT / "data/processed/corpus.jsonl").open(encoding="utf-8") as fh:
        corpus_text += " ".join(fold(json.loads(line)["answer"]) for line in fh) + " "
    fictional = agent["fictional_unanswerable_test"] + agent["fictional_unanswerable_dev"]
    # fictional condition names (capitalised words after the leading question word) must not occur in the corpus
    assert_absent([n for q in fictional for n in re.findall(r"\b[A-Z][a-z]+(?:-[A-Z][a-z]+)?", q)[1:]], corpus_text)
    for it in agent["absent_topic_test"]:
        assert_absent(it["terms"], corpus_text)
    c_items: list[EvaluationExample] = []
    c_items += personal_and_controls(corpus, agent["personal_templates_test"], "test", "c", rng_c, used)
    c_items += hard_negatives(corpus, 50, "test", "c", rng_c, used)
    c_items += simple_items(
        agent["fictional_unanswerable_test"],
        "c-fic",
        "test",
        case_type="unanswerable",
        note="unanswerable:fictional",
        behavior="abstain",
        reason="no_relevant_evidence",
    )
    c_items += simple_items(
        [it["q"] for it in agent["absent_topic_test"]],
        "c-oos",
        "test",
        case_type="unanswerable",
        note="unanswerable:out_of_corpus",
        behavior="abstain",
        reason="no_relevant_evidence",
    )
    c_items += simple_items(
        agent["ambiguous_test"],
        "c-amb",
        "test",
        case_type="ambiguous",
        note="ambiguous",
        behavior="either",
        reason=None,
    )
    c_items += fixture_items(agent)
    sets["test_track_c"] = c_items

    # DEV -----------------------------------------------------------------------------------------
    rng_d = _rng(31)
    used_d: set[str] = {fold(r.topic) for r in dev_recs}
    dev = track_a(corpus, dev_recs, agent["track_a_paraphrases"], "dev", "d")
    dev += personal_and_controls(corpus, agent["personal_templates_dev"], "validation", "d", rng_d, used_d)
    dev += simple_items(
        agent["fictional_unanswerable_dev"],
        "d-fic",
        "dev",
        case_type="unanswerable",
        note="unanswerable:fictional",
        behavior="abstain",
        reason="no_relevant_evidence",
    )
    dev += hard_negatives(corpus, 5, "validation", "d", rng_d, used_d)
    sets["dev"] = dev

    # Leakage over everything at once (cross-set duplicates are errors too) ------------------------
    all_items = [e for s in sets.values() for e in s]
    record_group = {r.record_id: r.split_group_id for r in corpus.records}
    group_split = {r.split_group_id: r.split for r in corpus.records}
    report = check_leakage(
        all_items,
        record_group,
        group_split,
        (r.question for r in corpus.records),
        fail=False,  # type: ignore[arg-type]
    )
    summary = {
        "builder_version": BUILDER_VERSION,
        "template_rule_version": TEMPLATE_RULE_VERSION,
        "seed": SEED,
        "split_version": meta.get("split_version"),
        "corpus_version": meta.get("corpus_version"),
        "agent_items_sha256": sha256_file(AGENT),
        "counts": {k: len(v) for k, v in sets.items()},
        "case_types": {k: dict(Counter(e.case_type for e in v)) for k, v in sets.items()},
        "notes_tags": {k: dict(Counter((e.notes or "").split(":")[0] for e in v)) for k, v in sets.items()},
        "question_provenance": {k: dict(Counter(e.question_provenance for e in v)) for k, v in sets.items()},
        "leakage": report.model_dump(),
        "protected_sets_checked": "none yet (SFT/classifier manifests not built); re-run check_leakage at E4 gate",
    }
    print(json.dumps({k: summary[k] for k in ("split_version", "counts", "case_types")}, indent=1))
    print("leakage errors:", len(report.errors), "warnings:", len(report.warnings))
    for line in (report.errors + report.warnings)[:30]:
        print("  ", line)
    if report.errors:
        return 1
    if args.check_only:
        return 0
    out = Path(args.out)
    meta_common = {
        k: summary[k]
        for k in (
            "builder_version",
            "template_rule_version",
            "seed",
            "split_version",
            "corpus_version",
            "agent_items_sha256",
        )
    }
    for name, items in sets.items():
        write_eval_set(items, out / f"{name}.jsonl", meta={**meta_common, "set": name})
    (out / "build_report.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    sheet = ROOT / "artifacts/evaluation/spotcheck/per_item/spotcheck_sheet.csv"
    sheet.parent.mkdir(parents=True, exist_ok=True)
    rows = spotcheck_rows(sets, corpus)
    with sheet.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out, "and", sheet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
