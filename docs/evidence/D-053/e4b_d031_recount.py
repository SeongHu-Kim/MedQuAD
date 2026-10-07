"""Step 1 re-count under the D-031 topic fold (read-only; counts only, no text, topics or record IDs printed).

Topic rule everywhere: medquad_qa.data.normalize.topic_key (NFKD, marks removed, casefold, non-alphanumeric
-> space). On-topic iff both keys are non-None and equal (an empty/missing topic never matches).

A. v2 build sft-mix-v2-20261007-035957: on-topic blocks per RAG prompt (train + validation builds).
B. (b') candidate availability per v2 train RAG question, with the metadata guards from both teammates,
   and per-guard drop counts: same split group, same topic key, then drop: same record, same question_type,
   gold's duplicate_group_id, boilerplate/excluded flags, whitespace-normalised answer equal to the gold's.
   (Sentinel question-type pair exclusions and length/overlap guards are not defined yet; not applied here.)
C. Serving retriever (frozen D-037 config from the environment) over validation questions, top 5:
   other on-topic blocks 0/1/2/3+ (all; gold retrieved), and for gold-retrieved questions whether each other
   on-topic block has the SAME or a DIFFERENT question_type than the question. Question types are looked up in
   the validation export only; an on-topic hit outside validation is counted separately (type not looked up).
Reads: build manifests, records_{train,validation}.jsonl (record_id, question, answer, topic, question_type,
quality_flags, split_group_id, duplicate_group_id), and the serving retriever's own artifacts.
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
from medquad_qa.data.normalize import topic_key  # noqa: E402
from medquad_qa.training.sft_data import EXCLUDED_FLAGS  # noqa: E402

BUILD = Path(sys.argv[1])
TOPK = 5
B = ("0", "1", "2", "3+")


def bucket(n):
    return "3+" if n >= 3 else str(n)


def fmt(c):
    n = sum(c.values())
    return f"(n={n}) " + ", ".join(f"{b}: {c[b]} ({c[b] / max(1, n):.1%})" for b in B)


def export(split):
    keep = ("record_id", "question", "answer", "topic", "question_type", "quality_flags", "split_group_id",
            "duplicate_group_id")
    out = {}
    for line in Path(f"data/processed/exports/records_{split}.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        x = {k: r.get(k) for k in keep}
        x["tkey"] = topic_key(x["topic"])
        x["norm_answer"] = " ".join((x["answer"] or "").split())
        out[x["record_id"]] = x
    return out


def on(a, b):
    return a is not None and a == b


def rows(name):
    return [json.loads(x) for x in (BUILD / name).open(encoding="utf-8") if x.strip()]


train, val = export("train"), export("validation")

# ---------------------------------------------------------------- A. training on-topic table (D-031)
print("== A. v2 build: on-topic blocks per RAG prompt (D-031 topic_key)")
for name, recs in (("sft_record_ids.jsonl", train), ("val_record_ids.jsonl", val)):
    c = Counter()
    for r in rows(name):
        if r["format"] == "closed_book":
            continue
        qk = recs[r["record_id"]]["tkey"]
        f = r["format"]
        c[f"{f}: n"] += 1
        c[f"{f}: question topic_key None"] += qk is None
        n_any = sum(on(recs[e]["tkey"], qk) for e in r["evidence_record_ids"])
        n_dis = sum(on(recs[e]["tkey"], qk) for e in r["evidence_record_ids"] if e != r["record_id"])
        c[f"{f}: >=1 on-topic block (incl. gold)"] += n_any > 0
        c[f"{f}: >=1 on-topic distractor (excl. gold)"] += n_dis > 0
    for k in sorted(c):
        print(f"  [{name}] {k}: {c[k]}")

# ---------------------------------------------------------------- B. (b') availability with guards
print("== B. (b') candidates per v2 train RAG question (same split group + same topic_key), guards applied in order")
by_group_topic: dict[tuple, list] = {}
for x in train.values():
    if x["tkey"] is not None:
        by_group_topic.setdefault((x["split_group_id"], x["tkey"]), []).append(x)
avail = {"rag_answerable": Counter(), "rag_insufficient": Counter()}
drops = {"rag_answerable": Counter(), "rag_insufficient": Counter()}
for r in rows("sft_record_ids.jsonl"):
    if r["format"] == "closed_book":
        continue
    q = train[r["record_id"]]
    cands = by_group_topic.get((q["split_group_id"], q["tkey"]), []) if q["tkey"] is not None else []
    d = drops[r["format"]]
    kept = 0
    for y in cands:
        if y["record_id"] == q["record_id"]:
            continue  # the question's own record (gold), not a candidate
        if y["question_type"] == q["question_type"]:
            d["same question_type"] += 1
        elif y["duplicate_group_id"] == q["duplicate_group_id"]:
            d["gold's duplicate group"] += 1
        elif EXCLUDED_FLAGS.intersection(y["quality_flags"] or []):
            d["boilerplate / excluded flags"] += 1
        elif y["norm_answer"] == q["norm_answer"]:
            d["answer text equal to gold's"] += 1
        else:
            kept += 1
    avail[r["format"]][bucket(kept)] += 1
for f in avail:
    print(f"  {f}: candidates after guards {fmt(avail[f])}")
    print(f"  {f}: drops by guard (first matching guard): {dict(drops[f])}")

# ---------------------------------------------------------------- C. serving retriever on validation
from medquad_qa.retrieval.factory import build_retriever  # noqa: E402

bundle = build_retriever()
ret = bundle.retriever
print("statuses:", [(s.name, s.ok, s.version) for s in bundle.statuses])
assert ret is not None, "serving retriever not built"
dist_all, dist_gold = Counter(), Counter()
qt_split = Counter()
per_q_types = Counter()
per_q_dup, per_q_same, per_q_diff = Counter(), Counter(), Counter()
other = Counter()
for x in val.values():
    hits, name, warnings = ret.retrieve_with_info(x["question"], TOPK)
    other[f"retriever name {name}"] += 1
    other["lexical_fallback warnings"] += any("lexical_fallback" in str(w) for w in warnings)
    other["question topic_key None"] += x["tkey"] is None
    others = [h for h in hits if h.record_id != x["record_id"] and on(topic_key(h.topic), x["tkey"])]
    dist_all[bucket(len(others))] += 1
    other["all questions: on-topic hits outside validation"] += sum(h.record_id not in val for h in others)
    if any(h.record_id == x["record_id"] for h in hits):
        dist_gold[bucket(len(others))] += 1
        dup = same = diff = outside = 0
        for h in others:
            y = val.get(h.record_id)
            if y is None:
                outside += 1
                qt_split["on-topic hit outside validation (type not looked up)"] += 1
            elif y["duplicate_group_id"] == x["duplicate_group_id"]:
                dup += 1
                qt_split["duplicate of gold (same duplicate_group_id)"] += 1
            elif y["question_type"] == x["question_type"]:
                same += 1
                qt_split["same question_type, different duplicate group"] += 1
            else:
                diff += 1
                qt_split["different question_type"] += 1
        per_q_types[f"dup>0={dup > 0} same>0={same > 0} diff>0={diff > 0}"] += 1
        per_q_dup[bucket(dup)] += 1
        per_q_same[bucket(same)] += 1
        per_q_diff[bucket(diff)] += 1
        other["gold-retrieved: on-topic hits outside validation"] += outside
print(f"== C. validation questions: {len(val)} | top_k={TOPK} | other on-topic blocks (excl. the question's own record)")
print(f"  all questions {fmt(dist_all)}")
print(f"  gold retrieved {fmt(dist_gold)}")
print(f"  gold retrieved, other on-topic blocks by question_type (block counts): {dict(qt_split)}")
print(f"  gold retrieved, per question (which categories present): {dict(per_q_types)}")
print(f"  gold retrieved, per question, duplicates of gold {fmt(per_q_dup)}")
print(f"  gold retrieved, per question, same type / different dup group {fmt(per_q_same)}")
print(f"  gold retrieved, per question, DIFFERENT question_type (what b' can reproduce) {fmt(per_q_diff)}")
for k in sorted(other):
    print(f"  {k}: {other[k]}")
