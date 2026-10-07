"""Answer-trained memorisation-probe subset (E4b amendment definition).

A train_probe item is answer-trained if gold_record_ids ∪ derived_from_record_ids contains a record that is the
`record_id` of a row with format closed_book or rag_answerable in the build's sft_record_ids.jsonl; role = that
row's format. Records appearing only in evidence_record_ids, or only as the question of a rag_insufficient row,
do not count. Usage: python probe_subset.py <sft_record_ids.jsonl> <train_probe.jsonl>  (prints counts only)
"""

import json
import sys
from collections import Counter

sft_path, probe_path = sys.argv[1], sys.argv[2]
rows = [json.loads(line) for line in open(sft_path, encoding="utf-8") if line.strip()]
target: dict[str, set[str]] = {}
evidence: set[str] = set()
insufficient_q: set[str] = set()
for r in rows:
    fmt = r.get("format")
    if fmt in ("closed_book", "rag_answerable"):
        target.setdefault(r["record_id"], set()).add(fmt)
    elif fmt == "rag_insufficient":
        insufficient_q.add(r["record_id"])
    evidence.update(r.get("evidence_record_ids") or [])
probe = [json.loads(line) for line in open(probe_path, encoding="utf-8") if line.strip()]
roles: Counter[str] = Counter()
for p in probe:
    ids = set(p["gold_record_ids"]) | set(p["derived_from_record_ids"])
    fm = set().union(*[target.get(i, set()) for i in ids])
    if fm == {"closed_book", "rag_answerable"}:
        roles["answer_trained_both_roles"] += 1
    elif fm == {"closed_book"}:
        roles["answer_trained_closed_book"] += 1
    elif fm == {"rag_answerable"}:
        roles["answer_trained_rag_gold"] += 1
    elif ids & evidence:
        roles["evidence_only"] += 1
    elif ids & insufficient_q:
        roles["rag_insufficient_question_only"] += 1
    else:
        roles["neither"] += 1
out = {
    "sft_rows": len(rows),
    "sft_formats": dict(Counter(r.get("format") for r in rows)),
    "split_versions": sorted({str(r.get("split_version")) for r in rows}),
    "probe_items": len(probe),
    "answer_trained_total": roles["answer_trained_closed_book"]
    + roles["answer_trained_rag_gold"]
    + roles["answer_trained_both_roles"],
    **{k: roles[k] for k in ("answer_trained_closed_book", "answer_trained_rag_gold", "answer_trained_both_roles",
                             "evidence_only", "rag_insufficient_question_only", "neither")},
}
print(json.dumps(out, indent=1, sort_keys=True))
