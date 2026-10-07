"""Topic-overlap check for the v2 build + Track C case-type counts (read-only; metadata and counts only).

Prints counts only: no topics, questions, answers, notes text or record IDs.
Reads: build manifests (IDs), records_{train,validation}.jsonl (only record_id and topic are used),
artifacts/evaluation/evalsets/test_track_c.jsonl and dev.jsonl (only case_type, answerable,
expected_abstention_reason, and the notes tag before the first ':' if it is a plain [a-z_] token).
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

OUT = Path(sys.argv[1])
TAG = re.compile(r"[a-z_]+")


def topics(split):
    t = {}
    for line in Path(f"data/processed/exports/records_{split}.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        t[r["record_id"]] = (r.get("topic") or "").casefold()  # same normalisation as the builder
    return t


def rows(name):
    return [json.loads(x) for x in (OUT / name).open(encoding="utf-8") if x.strip()]


print("== 1. on-topic blocks in v2 RAG prompts (topic match = case-insensitive exact, as in the builder)")
for name, split in (("sft_record_ids.jsonl", "train"), ("val_record_ids.jsonl", "validation")):
    top = topics(split)
    c = Counter()
    for r in rows(name):
        if r["format"] == "closed_book":
            continue
        qt = top[r["record_id"]]
        c[f"{r['format']}: n"] += 1
        c[f"{r['format']}: question topic empty"] += qt == ""
        ev = r["evidence_record_ids"]
        on_any = sum(top[e] == qt for e in ev)
        on_distr = sum(top[e] == qt for e in ev if e != r["record_id"])
        c[f"{r['format']}: >=1 on-topic block (any, incl. gold)"] += on_any > 0
        c[f"{r['format']}: >=1 on-topic distractor (excl. gold)"] += on_distr > 0
        c[f"{r['format']}: on-topic distractors total"] += on_distr
    for k in sorted(c):
        print(f"[{name}] {k}: {c[k]}")

print("== 2. eval-set case types (counts only)")
for es in ("test_track_c", "dev"):
    c = Counter()
    for line in open(f"artifacts/evaluation/evalsets/{es}.jsonl", encoding="utf-8"):
        e = json.loads(line)
        notes = e.get("notes") or ""
        head = notes.split(":", 1)[0]
        tag = head if TAG.fullmatch(head) else ("<untagged>" if not notes else "<non-tag notes>")
        c[(e["case_type"], tag, e["answerable"], e.get("expected_abstention_reason"))] += 1
    print(f"[{es}] (case_type, notes tag, answerable, expected_abstention_reason): count")
    for k, v in sorted(c.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        print(f"  {k}: {v}")
