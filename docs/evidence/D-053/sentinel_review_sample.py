"""Write the 40-item sentinel review file for the user (prints only path, row count and SHA-256; never text).

Population: sentinel_same_topic rows of artifacts/models/builds/sft-mix-v2b-20261007-083619/sft_record_ids.jsonl.
Stratum: question-type pair = (sentinel question's question_type -> question_type of its FIRST on-topic block in
prompt order, i.e. the highest-ranked same-topic block, since distractors are ordered by dense score).
Allocation: 40 slots proportional to stratum size, largest-remainder rounding, ties broken by stratum name; then
within each stratum a seeded random sample (random.Random(SEED + ':' + stratum)).
Each prompt is rebuilt exactly from the manifest: the question and every evidence block E1..Ek from
evidence_chunk_ids, rendered with the pipeline's own build_rag_messages (the user message the model sees, which
neutralises evidence text exactly as at train/serve time), followed by a per-block table (label, on-topic flag,
block question_type). Reads only the build manifest and records_train.jsonl (train split).
Output: ~/medquad_review/sentinel_review_v2b_<SEED>.md, mode 0600, never overwritten (O_EXCL).
"""

import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")
from medquad_qa.rag.prompts import EvidenceBlock, build_rag_messages  # noqa: E402
from medquad_qa.retrieval.corpus import chunk_record  # noqa: E402
from medquad_qa.training.sft_data import read_records  # noqa: E402

SEED = 20261007
N = 40
BUILD = Path("artifacts/models/builds/sft-mix-v2b-20261007-083619")
OUT_DIR = Path.home() / "medquad_review"
OUT = OUT_DIR / f"sentinel_review_v2b_{SEED}.md"

rows = [json.loads(x) for x in (BUILD / "sft_record_ids.jsonl").open(encoding="utf-8") if x.strip()]
pop = [r for r in rows if r.get("sentinel_kind") == "sentinel_same_topic"]
recs = {r.record_id: r for r in read_records(Path("data/processed/exports/records_train.jsonl"))}


def stratum(r: dict) -> str:
    first_on = next(i for i, f in enumerate(r["evidence_on_topic"]) if f)
    return f"{r['question_type']}->{recs[r['evidence_record_ids'][first_on]].question_type}"


strata: dict[str, list] = defaultdict(list)
for r in pop:
    strata[stratum(r)].append(r)
total = len(pop)
quota = {s: N * len(v) / total for s, v in strata.items()}
alloc = {s: int(q) for s, q in quota.items()}
for s in sorted(quota, key=lambda s: (-(quota[s] - alloc[s]), s))[: N - sum(alloc.values())]:
    alloc[s] += 1
picked = []
for s in sorted(strata):
    if alloc[s]:
        picked += random.Random(f"{SEED}:{s}").sample(sorted(strata[s], key=lambda r: r["record_id"]), alloc[s])
assert len(picked) == N, len(picked)
random.Random(f"{SEED}:order").shuffle(picked)

chunk_cache: dict[str, dict[str, object]] = {}


def chunk(cid: str):
    rid = cid.split("#", 1)[0]
    if rid not in chunk_cache:
        chunk_cache[rid] = {c.chunk_id: c for c in chunk_record(recs[rid])}
    return chunk_cache[rid][cid]


parts = [
    f"# Sentinel review, sft-mix-v2b-20261007-083619 (seed {SEED})\n",
    "For each sample: does ANY evidence block answer or partly answer the question?\n"
    "Fill the VERDICT line with exactly one of: no | partial | yes\n"
    "Rule (fixed in advance): PASS if (partial + yes) <= 4 of 40.\n",
    f"Population: {total} same-topic sentinel prompts; strata (question type -> first on-topic block type): "
    + ", ".join(f"{s}={alloc[s]}/{len(strata[s])}" for s in sorted(strata) if alloc[s]) + "\n",
]
for i, r in enumerate(picked, 1):
    q = recs[r["record_id"]]
    blocks = []
    for lab, (rid, cid) in enumerate(zip(r["evidence_record_ids"], r["evidence_chunk_ids"], strict=True), 1):
        er, c = recs[rid], chunk(cid)
        blocks.append(EvidenceBlock(label=f"E{lab}", text=c.text, topic=er.topic, source=er.source_name))
    user_msg = build_rag_messages(q.question, blocks)[-1].content
    table = "\n".join(
        f"| E{lab} | {'ON-TOPIC' if on else 'off-topic'} | {recs[rid].question_type} |"
        for lab, (rid, on) in enumerate(zip(r["evidence_record_ids"], r["evidence_on_topic"], strict=True), 1)
    )
    parts.append(
        f"\n---\n## Sample {i} of {N}  (stratum {stratum(r)}; question type {q.question_type}; "
        f"on-topic blocks {r['on_topic_total']})\n\n```text\n{user_msg}\n```\n\n"
        f"| block | on-topic | block question type |\n|---|---|---|\n{table}\n\nVERDICT: \n"
    )
data = "".join(parts).encode("utf-8")

OUT_DIR.mkdir(mode=0o700, exist_ok=True)
fd = os.open(OUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)  # refuses to overwrite
with os.fdopen(fd, "wb") as f:
    f.write(data)
print(f"path: {OUT}")
print(f"rows: {N} (VERDICT lines: {data.count(b'\\nVERDICT: ')})")
print(f"sha256: {hashlib.sha256(data).hexdigest()}")
print(f"mode: {oct(os.stat(OUT).st_mode & 0o777)}")
