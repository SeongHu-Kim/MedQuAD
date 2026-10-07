"""E4b step-2 follow-up (read-only, counts only; never prints record text).

1. Dropped evidence: gold kept? which positions dropped? (intended gold slot reconstructed from the
   build's seeded RNG and verified against all non-dropped rows).
2. Answerable targets: recomputed in memory with the build's own cited_target() + Qwen tokenizer
   (tokenizer files only, no weights); markers, sentence ends, first-marker token position.
3. train_probe: role of each probe item's record(s) in the v2 build.
"""

import json
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
from medquad_qa.retrieval.corpus import chunk_record  # noqa: E402
from medquad_qa.training.sft_data import read_records  # noqa: E402
from medquad_qa.training.sft_rag_data import cited_target  # noqa: E402

OUT = Path(sys.argv[1])
K, SEED, CAP = 5, 20261006, 256
MARK = re.compile(r"\[E[1-5]\]")
# final piece (trailing " [E#]" removed) ends in a common abbreviation; checked in this order, first match wins
ABBR = [
    ("e.g.", re.compile(r"\be\.g\.$", re.I)),
    ("i.e.", re.compile(r"\bi\.e\.$", re.I)),
    ("etc.", re.compile(r"\betc\.$", re.I)),
    ("vs.", re.compile(r"\bvs\.$", re.I)),
    ("Dr.", re.compile(r"\bDr\.$")),
    ("single capital + period", re.compile(r"(?:^|[\s(])[A-Z]\.$")),
]
ONLY_ABBR = "--abbr-only" in sys.argv


def rows(name):
    return [json.loads(x) for x in (OUT / name).open(encoding="utf-8") if x.strip()]


def p90(xs):
    s = sorted(xs)
    return s[int(0.9 * (len(s) - 1))] if s else None


# ---------------------------------------------------------------- 1. dropped evidence
def drops(name):
    rs = [r for r in rows(name) if r["format"] != "closed_book"]
    rng = random.Random(f"{SEED}:mix")  # same construction as MixedSFTBuilder.build
    ok_recon = bad_recon = 0
    res = Counter()
    n_dropped_dist = Counter()
    shift = Counter()
    missing_gold = Counter()  # (blocks dropped, intended slot) for answerable rows whose gold ID is absent
    for r in rs:
        ev = r["evidence_record_ids"]
        final = intended = None
        if r["format"] == "rag_answerable":
            intended = rng.randrange(K)  # 0-based slot; drawn once per answerable example
            final = ev.index(r["record_id"]) if r["record_id"] in ev else None
            if final is None:
                missing_gold[(K - len(ev), f"E{intended + 1}")] += 1
            elif len(ev) == K:
                ok_recon += final == intended
                bad_recon += final != intended
        if len(ev) < K:
            n = K - len(ev)
            n_dropped_dist[n] += 1
            if r["format"] == "rag_answerable":
                res["drop_answerable_gold_kept" if final is not None else "drop_answerable_gold_MISSING"] += 1
                if final is not None:
                    shift["gold slot unchanged" if final == intended else f"gold moved E{intended + 1}->E{final + 1}"] += 1
            else:
                res["drop_insufficient"] += 1
    flag = "  <-- FLAG: mismatch > 0" if bad_recon else ""
    print(f"[{name}] RNG reconstruction on non-dropped answerable rows: match={ok_recon} mismatch={bad_recon}{flag}")
    print(f"[{name}] drop cases: {sum(n_dropped_dist.values())} | {dict(res)} | blocks dropped per case: {dict(sorted(n_dropped_dist.items()))}")
    print(f"[{name}] gold position in drop cases: {dict(shift)}")
    print(f"[{name}] answerable rows with gold ID missing (any row): {sum(missing_gold.values())}"
          f" | by (blocks dropped, intended slot): {dict(missing_gold)}")


if not ONLY_ABBR:
    print("== 1. dropped evidence (dropped blocks are always the lowest-ranked distractors: picked.pop())")
    drops("sft_record_ids.jsonl")
    drops("val_record_ids.jsonl")

# ---------------------------------------------------------------- 2. answerable targets
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507", revision="cdbee75f17c01a7cc42f958dc650907174af0554", local_files_only=True
)


def ntok(t):
    return len(tok.encode(t, add_special_tokens=False))


def targets(name, split):
    recs = {r.record_id: r for r in read_records(Path(f"data/processed/exports/records_{split}.jsonl"))}
    rs = [r for r in rows(name) if r["format"] == "rag_answerable"]
    stats = {True: Counter(), False: Counter()}
    abbr = {True: Counter(), False: Counter()}
    first_full, first_kept = [], []
    agree = disagree = gold_missing = full_no_marker = 0
    for r in rs:
        if r["record_id"] not in r["evidence_record_ids"]:
            gold_missing += 1
            continue
        gold = chunk_record(recs[r["record_id"]])[0]  # builder: first_chunk[...] = chunk_record(record)[0]
        label = f"E{r['evidence_record_ids'].index(r['record_id']) + 1}"
        kept, used, total = cited_target(gold.text, label, lambda t: ntok(t) <= CAP)
        full, _, _ = cited_target(gold.text, label, lambda t: True)
        recomputed_trunc = used < total
        if recomputed_trunc == r["truncated"]:
            agree += 1
        else:
            disagree += 1  # prompt budget tighter than the 256 cap for this row (budget-limited)
        s = stats[r["truncated"]]
        s["n"] += 1
        s["kept has >=1 marker" if MARK.search(kept) else "kept has NO marker"] += 1
        last = MARK.sub("", kept.splitlines()[-1] if kept else "").rstrip()
        s["last sentence ends . ! ?" if last.endswith((".", "!", "?")) else "last piece has no terminal punctuation"] += 1
        abbr[r["truncated"]][next((n for n, rx in ABBR if rx.search(last)), "no abbreviation ending")] += 1
        m = MARK.search(full)
        if m:
            first_full.append(ntok(full[: m.start()]))
        else:
            full_no_marker += 1
        m2 = MARK.search(kept)
        if m2:
            first_kept.append(ntok(kept[: m2.start()]))
    for t in (True, False):
        print(f"[{name}] final-piece abbreviation endings, truncated={t}: {dict(abbr[t])}")
    if ONLY_ABBR:
        return
    print(f"[{name}] answerable rows skipped because gold ID missing from evidence: {gold_missing}")
    print(f"[{name}] recomputed truncation flag agrees with build manifest: {agree}/{agree + disagree} (disagree={disagree})")
    for t in (True, False):
        print(f"[{name}] truncated={t}: {dict(stats[t])}")
    print(f"[{name}] full targets with no marker (excluded from stats): {full_no_marker}")
    print(f"[{name}] first-marker token position, full targets: median={statistics.median(first_full)} p90={p90(first_full)} max={max(first_full)}")
    print(f"[{name}] first-marker token position, kept targets: median={statistics.median(first_kept)} p90={p90(first_kept)}")


print("== 2. answerable targets (EOS: code appends eot_ids to every RAG target, truncated or not)")
targets("sft_record_ids.jsonl", "train")
targets("val_record_ids.jsonl", "validation")
if ONLY_ABBR:
    sys.exit(0)

# ---------------------------------------------------------------- 3. train_probe roles
QROLE = {"closed_book": "closed_book target", "rag_answerable": "rag_answerable gold target",
         "rag_insufficient": "rag_insufficient question (answer not trained)"}
roles: dict[str, set[str]] = {}
for r in rows("sft_record_ids.jsonl"):
    roles.setdefault(r["record_id"], set()).add(QROLE[r["format"]])
    for e in r["evidence_record_ids"]:
        if e != r["record_id"]:  # the gold block is the target itself, already counted above
            roles.setdefault(e, set()).add("evidence block in another example (loss-masked prompt)")
probe = [json.loads(x) for x in open("artifacts/evaluation/evalsets/train_probe.jsonl", encoding="utf-8")]
by = Counter()
for p in probe:
    ids = sorted(set(p["gold_record_ids"]) | set(p["derived_from_record_ids"]))
    rs = sorted(set().union(*(roles.get(i, {"absent from v2 build"}) for i in ids)))
    by[" + ".join(rs)] += 1
print("== 3. train_probe items (n=%d) by v2 role of their gold/derived record IDs:" % len(probe))
for k_, v in by.most_common():
    print(f"  {v:3d}  {k_}")
