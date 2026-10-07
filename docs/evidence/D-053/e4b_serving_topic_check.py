"""Serving-side on-topic rate on VALIDATION questions + (b') availability on TRAIN (read-only; counts only).

Section 1: the frozen serving retriever (D-037: dense_fallback on question_answer, index dense-qa-dc6b6a345fca,
top_k 5), built by the serving factory from the environment, CPU, offline, over every validation-export
question. Topic rule = builder rule (case-insensitive exact). On-topic uses hit.topic from the serving store.
Section 2: for each v2 train RAG question, how many same-topic records with a DIFFERENT question_type exist
in the train export (the pool option (b') would draw from); metadata fields only.
Prints counts only: no questions, answers, topics or record IDs.
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
from medquad_qa.retrieval.factory import build_retriever  # noqa: E402
from medquad_qa.training.sft_data import EXCLUDED_FLAGS  # noqa: E402

BUILD = Path(sys.argv[1])
TOPK = 5


def bucket(n: int) -> str:
    return "3+" if n >= 3 else str(n)


def export(split: str) -> list[dict]:
    keep = ("record_id", "question", "topic", "question_type", "quality_flags", "split_group_id")
    out = []
    for line in Path(f"data/processed/exports/records_{split}.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        out.append({k: r.get(k) for k in keep})
    return out


# ------------------------------------------------------------------ 1. serving retriever on validation
bundle = build_retriever()
r = bundle.retriever
print("statuses:", [(s.name, s.ok, s.version) for s in bundle.statuses])
assert r is not None, "serving retriever not built"
val = export("validation")
val_ids = {x["record_id"] for x in val}
dist = {"all": Counter(), "gold retrieved": Counter(), "gold retrieved: on-topic NON-gold blocks": Counter()}
other = Counter()
for x in val:
    hits, name, warnings = r.retrieve_with_info(x["question"], TOPK)
    other[f"retriever name {name}"] += 1
    other["lexical_fallback warnings"] += any("lexical_fallback" in str(w) for w in warnings)
    other["hits outside validation split"] += sum(h.record_id not in val_ids for h in hits)
    other["hits returned"] += len(hits)
    qt = (x["topic"] or "").casefold()
    other["question topic empty"] += qt == ""
    on = [(h.topic or "").casefold() == qt for h in hits]
    dist["all"][bucket(sum(on))] += 1
    gold = [h.record_id == x["record_id"] for h in hits]
    if any(gold):
        dist["gold retrieved"][bucket(sum(on))] += 1
        dist["gold retrieved: on-topic NON-gold blocks"][bucket(sum(o and not g for o, g in zip(on, gold)))] += 1
    other["eligible as builder question (no excluded flags)"] += not EXCLUDED_FLAGS.intersection(x["quality_flags"] or [])
print(f"== 1. validation questions: {len(val)} | top_k={TOPK}")
for k, c in dist.items():
    n = sum(c.values())
    print(f"  on-topic blocks in top {TOPK} [{k}] (n={n}): " + ", ".join(f"{b}: {c[b]} ({c[b] / max(1, n):.1%})" for b in ("0", "1", "2", "3+")))
for k in sorted(other):
    print(f"  {k}: {other[k]}")

# ------------------------------------------------------------------ 2. (b') pool availability on train
train = export("train")
by_topic: dict[str, list[dict]] = {}
for x in train:
    if not EXCLUDED_FLAGS.intersection(x["quality_flags"] or []):
        by_topic.setdefault((x["topic"] or "").casefold(), []).append(x)
tr = {x["record_id"]: x for x in train}
rows = [json.loads(line) for line in (BUILD / "sft_record_ids.jsonl").open(encoding="utf-8") if line.strip()]
avail = {"rag_answerable": Counter(), "rag_insufficient": Counter()}
for row in rows:
    if row["format"] == "closed_book":
        continue
    q = tr[row["record_id"]]
    same = [
        y for y in by_topic.get((q["topic"] or "").casefold(), [])
        if y["record_id"] != q["record_id"] and y["question_type"] != q["question_type"]
        and y["split_group_id"] == q["split_group_id"]
    ]
    avail[row["format"]][bucket(len(same))] += 1
print("== 2. (b') candidates per v2 train RAG question: same-topic, different question_type, same split group")
for f, c in avail.items():
    n = sum(c.values())
    print(f"  {f} (n={n}): " + ", ".join(f"{b}: {c[b]} ({c[b] / max(1, n):.1%})" for b in ("0", "1", "2", "3+")))
