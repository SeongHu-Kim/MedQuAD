"""Build the frozen evaluation sets (E2) from the signed-off split.

Deterministic: every random choice uses ``random.Random(SEED)`` over sorted inputs. Outputs hold IDs,
paraphrased/agent-written questions and labels only (D-015); no dataset answer text.

Sets:
- ``test_track_a``: 300 TEST queries, <=1 per test split group. 200 template paraphrases (``synthetic_rule``)
  + 100 agent-written questions (``llm_generated_unreviewed``). Gold = non-boilerplate records with the same
  folded topic and question_type. Also the Track B query set (same queries, all four modes).
- ``train_probe``: 100 template paraphrases of TRAIN records (Track B memorization probe; never pooled with TEST).
- ``test_track_c``: ~240 robustness items on TEST topics or out-of-corpus/fictional topics.
- ``dev``: 30 Track-A-style + 30 Track-C-style items from VALIDATION groups; the only set allowed for tuning.

Agent-written text lives in ``configs/evaluation/agent_items.json`` (written by the evaluation agent, an AI;
labelled ``llm_generated_unreviewed`` and never ``human_reviewed``).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medquad_qa.contracts.evaluation import EvaluationExample

SEED = 20261006
BUILDER_VERSION = "evalset-builder-v1"
TEMPLATE_RULE_VERSION = "paraphrase-templates-v1"
EXCLUDED_FLAGS = frozenset({"boilerplate_answer", "malformed_question", "short_answer", "missing_topic"})

# Paraphrase templates per question_type. None reuses MedQuAD's own phrasing ("What is (are)", "What are the
# symptoms of", ...); all are checked against every indexed question by check_leakage (E3 + near-copy warning).
TEMPLATES: dict[str, tuple[str, ...]] = {
    "information": (
        "Can you give me an overview of {t}?",
        "I'd like a plain-language explanation of {t}.",
        "Explain {t} in simple terms.",
    ),
    "symptoms": (
        "How does {t} usually show itself, and what signs should someone watch for?",
        "Which signs and symptoms are typical of {t}?",
        "What features or complaints are commonly seen with {t}?",
    ),
    "treatment": (
        "How is {t} usually managed or treated?",
        "Which therapies are available for {t}?",
        "What options exist for managing {t}?",
    ),
    "inheritance": (
        "Can {t} be passed down from parents to their children?",
        "How is {t} transmitted within families?",
        "Does {t} run in families, and if so, in what pattern?",
    ),
    "frequency": (
        "How common is {t}?",
        "Roughly how often does {t} occur in the population?",
        "Is {t} rare, and about how many people have it?",
    ),
    "genetic_changes": (
        "Which gene mutations are responsible for {t}?",
        "What is known about the genetic basis of {t}?",
        "Mutations in which genes lead to {t}?",
    ),
    "causes": (
        "What leads to {t}?",
        "Why does {t} develop?",
        "What is the underlying cause of {t}?",
    ),
    "exams_and_tests": (
        "How do doctors confirm a diagnosis of {t}?",
        "Which tests are used to detect {t}?",
        "How is {t} identified by clinicians?",
    ),
    "outlook": (
        "What is the long-term prognosis for people with {t}?",
        "How does {t} typically progress over time?",
        "What can people with {t} expect in the long run?",
    ),
    "research": (
        "Are there ongoing studies or clinical trials on {t}?",
        "What are researchers currently investigating about {t}?",
        "Is any new research being done on {t}?",
    ),
    "considerations": (
        "Are there practical tips for living with {t}?",
        "What everyday steps can help someone with {t}?",
        "What should people with {t} keep in mind day to day?",
    ),
    "susceptibility": (
        "Which people are more likely to develop {t}?",
        "What raises someone's chance of getting {t}?",
        "Who has a higher risk of {t}?",
    ),
    "prevention": (
        "Can {t} be avoided, and how?",
        "What steps lower the chance of getting {t}?",
        "Is it possible to prevent {t}?",
    ),
    "stages": (
        "How is {t} divided into stages?",
        "What staging system is used for {t}?",
        "What are the different phases of {t}?",
    ),
    "complications": (
        "What problems can {t} lead to?",
        "Which complications are linked to {t}?",
        "What other health issues can arise from {t}?",
    ),
}

# Question types asked about a topic that LACKS them in the corpus (Track C hard negatives).
HARD_NEGATIVE_QTYPES = ("frequency", "genetic_changes", "research", "stages", "complications", "prevention")


def fold(text: str) -> str:
    t = unicodedata.normalize("NFKD", text)
    t = "".join(c for c in t if not unicodedata.combining(c)).casefold()
    return re.sub(r"[^0-9a-z]+", " ", t).strip()


def stable_id(prefix: str, *parts: str) -> str:
    return f"{prefix}-" + hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class Rec:
    record_id: str
    question: str
    topic: str
    question_type: str
    split: str
    split_group_id: str
    source: str
    flags: frozenset[str]


class Corpus:
    def __init__(self, records: Sequence[Rec]) -> None:
        self.records = list(records)
        self.by_id = {r.record_id: r for r in self.records}
        self.by_topic_qtype: dict[tuple[str, str], list[Rec]] = defaultdict(list)
        self.qtypes_by_topic: dict[str, set[str]] = defaultdict(set)
        self.split_of_topic: dict[str, set[str]] = defaultdict(set)
        for r in self.records:
            k = fold(r.topic)
            self.by_topic_qtype[(k, r.question_type)].append(r)
            self.qtypes_by_topic[k].add(r.question_type)
            self.split_of_topic[k].add(r.split)

    def gold(self, rec: Rec) -> list[str]:
        return sorted(
            r.record_id
            for r in self.by_topic_qtype[(fold(rec.topic), rec.question_type)]
            if "boilerplate_answer" not in r.flags
        )

    @classmethod
    def load(cls, root: Path) -> tuple[Corpus, dict[str, Any]]:
        manifest = {}
        with (root / "data/manifests/split_manifest.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                m = json.loads(line)
                manifest[m["record_id"]] = m
        recs = []
        with (root / "data/processed/corpus.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                c = json.loads(line)
                m = manifest[c["record_id"]]
                recs.append(
                    Rec(
                        record_id=c["record_id"],
                        question=c["question"],
                        topic=c["topic"] or "",
                        question_type=c["question_type"] or "other",
                        split=m["split"],
                        split_group_id=m["split_group_id"],
                        source=m["source"],
                        flags=frozenset(m["quality_flags"]),
                    )
                )
        meta = json.loads((root / "data/manifests/split_manifest.meta.json").read_text(encoding="utf-8"))
        return cls(recs), meta


def eligible(r: Rec) -> bool:
    return bool(r.topic) and r.question_type in TEMPLATES and not (r.flags & EXCLUDED_FLAGS)


def pick_one_per_group(corpus: Corpus, split: str, n: int, rng: random.Random, exclude_groups: set[str]) -> list[Rec]:
    """<=1 eligible record per split group, balancing question types (round-robin over a shuffled type order)."""
    groups: dict[str, list[Rec]] = defaultdict(list)
    for r in corpus.records:
        if r.split == split and eligible(r) and r.split_group_id not in exclude_groups and corpus.gold(r):
            groups[r.split_group_id].append(r)
    keys = sorted(groups)
    rng.shuffle(keys)
    type_counts: dict[str, int] = defaultdict(int)
    chosen: list[Rec] = []
    for g in keys:
        if len(chosen) == n:
            break
        cands = sorted(groups[g], key=lambda r: (type_counts[r.question_type], r.record_id))
        best = [c for c in cands if type_counts[c.question_type] == type_counts[cands[0].question_type]]
        pick = best[rng.randrange(len(best))]
        type_counts[pick.question_type] += 1
        chosen.append(pick)
    if len(chosen) < n:
        raise ValueError(f"only {len(chosen)} eligible groups in {split}, need {n}")
    return chosen


def template_question(rec: Rec, rng: random.Random) -> tuple[str, int]:
    opts = TEMPLATES[rec.question_type]
    i = rng.randrange(len(opts))
    return opts[i].format(t=rec.topic), i


def track_a_example(
    rec: Rec, corpus: Corpus, question: str, *, agent: bool, eval_split: str, method: str, prefix: str
) -> EvaluationExample:
    return EvaluationExample(
        example_id=stable_id(prefix, rec.record_id, method),
        question=question,
        gold_record_ids=corpus.gold(rec),
        answerable=True,
        expected_behavior="answer",
        label_provenance="synthetic_rule",
        question_provenance="llm_generated_unreviewed" if agent else "synthetic_rule",
        paraphrase_method=method,
        reviewer="none",
        derived_from_record_ids=[rec.record_id],
        split_group_id=rec.split_group_id,
        eval_split=eval_split,
        evaluation_track="finetune_generalization" if eval_split == "train_probe" else "corpus_grounded",
        case_type="answerable",
        question_type=rec.question_type,
        topic=rec.topic,
    )


def assert_absent(terms: Iterable[str], corpus_text_folded: str) -> None:
    for t in terms:
        if f" {fold(t)} " in corpus_text_folded:
            raise ValueError(f"out-of-corpus term {t!r} occurs in the corpus")
