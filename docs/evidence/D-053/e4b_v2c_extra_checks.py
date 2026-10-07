"""v2c extra checks (read-only; build manifests only = IDs/flags/counts; prints counts only).

1. Every sentinel row (format rag_insufficient) has on_topic_total == 0 and no on-topic evidence flag.
2. Answerable rows match v2b: for train and validation, the rag_answerable rows of the v2c build equal those of
   sft-mix-v2b-20261007-083619, compared as whole JSON rows with only 'mix_version' removed if present.
3. Sentinel kinds and closed-book count, for the record.
Usage: e4b_v2c_extra_checks.py <v2c_build_dir> <v2b_build_dir>
"""

import json
import sys
from collections import Counter
from pathlib import Path

c_dir, b_dir = Path(sys.argv[1]), Path(sys.argv[2])
fail = []


def rows(d: Path, name: str) -> list[dict]:
    return [json.loads(x) for x in (d / name).open(encoding="utf-8") if x.strip()]


def strip(r: dict) -> dict:
    return {k: v for k, v in r.items() if k != "mix_version"}


for name in ("sft_record_ids.jsonl", "val_record_ids.jsonl"):
    c_rows, b_rows = rows(c_dir, name), rows(b_dir, name)
    sen = [r for r in c_rows if r["format"] == "rag_insufficient"]
    bad = sum(r["on_topic_total"] != 0 or any(r["evidence_on_topic"]) for r in sen)
    print(f"[{name}] sentinels: {len(sen)} | with any on-topic block: {bad} | kinds: {dict(Counter(r.get('sentinel_kind') for r in sen))}")
    if bad:
        fail.append(f"{name}: {bad} sentinels with on-topic blocks")
    ca = [strip(r) for r in c_rows if r["format"] == "rag_answerable"]
    ba = [strip(r) for r in b_rows if r["format"] == "rag_answerable"]
    same = ca == ba
    diff_ids = sum(x != y for x, y in zip(ca, ba)) if len(ca) == len(ba) else None
    print(f"[{name}] answerable rows: v2c {len(ca)} | v2b {len(ba)} | identical (whole rows, order kept): {same}"
          + ("" if same else f" | differing rows: {diff_ids}"))
    if not same:
        fail.append(f"{name}: answerable rows differ from v2b")
    print(f"[{name}] closed-book rows: v2c {sum(r['format'] == 'closed_book' for r in c_rows)}"
          f" | v2b {sum(r['format'] == 'closed_book' for r in b_rows)}")
print("EXTRA CHECKS:", "OK" if not fail else f"FAIL {fail}")
sys.exit(1 if fail else 0)
