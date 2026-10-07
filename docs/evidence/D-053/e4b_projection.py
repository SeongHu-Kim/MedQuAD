"""Step-2 projections for sft-mix-v2b (read-only; counts only, no text, topics or record IDs printed).

P1. Serving (frozen D-037 retriever from the environment, CPU, top 5) on validation questions with gold
    retrieved: per question, number of OTHER on-topic blocks of a DIFFERENT question_type, buckets 0..4
    (splits the earlier '3+'). D-031 topic_key; question types from the validation export only.
P2. Sentinel candidate availability per question type in the TRAIN pool under G1-G4 + G7 (metadata only):
    candidates = same split group, same topic_key, record != question; drop G1 same question_type,
    G2 gold's duplicate group, G3 EXCLUDED_FLAGS, G4 whitespace-normalised ANSWER equal to the question's
    (record-level proxy for the chunk-level G4), G7 model-engineer's pair table. G5/G6 need chunk text and
    scores and are NOT applied, so availability here is an upper bound.
    Reported: zero-candidate share per question type over the whole eligible train pool; then for the two
    sentinel populations: (a) table as proposed (all types), (b) information/other/support_groups removed.
P3. Sentinel draw = 450 allowed-type + 50 information-type (off-topic evidence only): can they be drawn from the
    builder's seeded order after the first 2,700 answerable draws; zero-candidate share of the 450; type mix of
    the 450 vs the hard-negative question types AS DEFINED IN CODE (evalset_builder.HARD_NEGATIVE_QTYPES) and the
    DEV hard negatives' missing types (fixed 'hard_negative:missing_qtype=<t>' notes tag only).
    No TEST eval set (Track C) is read.
Reads: records_{train,validation}.jsonl (record_id, question, answer, topic, question_type, quality_flags,
split_group_id, duplicate_group_id); dev.jsonl (notes tag only); serving artifacts.
"""

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
from medquad_qa.data.normalize import topic_key  # noqa: E402
from medquad_qa.evaluation.evalset_builder import HARD_NEGATIVE_QTYPES  # noqa: E402
from medquad_qa.training.sft_data import EXCLUDED_FLAGS  # noqa: E402

SEED, N_ANS, N_SENT, N_INFO_SENT, TOPK = 20261006, 2700, 500, 50, 5
NO_SENTINEL = {"information", "other", "support_groups"}
INFO = "information"
G7 = {  # sentinel question type -> excluded block question types (model-engineer's table)
    "information": "ALL", "other": "ALL", "support_groups": "ALL",
    "causes": {"genetic_changes", "inheritance", "susceptibility", INFO},
    "genetic_changes": {"causes", "inheritance", "susceptibility", INFO},
    "inheritance": {"genetic_changes", "causes", "susceptibility", INFO},
    "susceptibility": {"causes", "genetic_changes", "inheritance", "frequency", INFO},
    "treatment": {"prevention", "considerations", "research", INFO},
    "prevention": {"treatment", "considerations", "causes", "susceptibility", INFO},
    "considerations": {"treatment", "prevention", INFO},
    "research": {"treatment", INFO},
    "symptoms": {"complications", "stages", "exams_and_tests", "outlook", INFO},
    "exams_and_tests": {"symptoms", "stages", INFO},
    "complications": {"symptoms", "outlook", "stages", INFO},
    "stages": {"symptoms", "complications", "outlook", "exams_and_tests", INFO},
    "outlook": {"complications", "stages", INFO},
    "frequency": {"susceptibility", INFO},
}
HN = re.compile(r"hard_negative:missing_qtype=([a-z_]+)")


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


def share_table(c_all, c_zero, order):
    for t in order:
        n = c_all[t]
        if n:
            print(f"    {t}: n={n} zero-candidate={c_zero[t]} ({c_zero[t] / n:.1%})")


train, val = export("train"), export("validation")
pool = [x for x in train.values() if not EXCLUDED_FLAGS.intersection(x["quality_flags"] or []) and (x["answer"] or "").strip()]

# ---------------------------------------------------------------- P2. sentinel availability under G1-G4 + G7
groups: dict[tuple, list] = {}
for x in train.values():
    if x["tkey"] is not None:
        groups.setdefault((x["split_group_id"], x["tkey"]), []).append(x)


def n_candidates(q):
    if q["tkey"] is None:
        return 0
    excl = G7.get(q["question_type"], set())
    if excl == "ALL":
        return 0
    k = 0
    for y in groups.get((q["split_group_id"], q["tkey"]), []):
        if (y["record_id"] == q["record_id"] or y["question_type"] == q["question_type"]
                or y["duplicate_group_id"] == q["duplicate_group_id"]
                or EXCLUDED_FLAGS.intersection(y["quality_flags"] or [])
                or y["norm_answer"] == q["norm_answer"] or y["question_type"] in excl):
            continue
        k += 1
    return k


ncand = {x["record_id"]: n_candidates(x) for x in pool}
c_all, c_zero = Counter(), Counter()
for x in pool:
    c_all[x["question_type"]] += 1
    c_zero[x["question_type"]] += ncand[x["record_id"]] == 0
order = [t for t, _ in c_all.most_common()]
print("== P2. sentinel candidates under G1-G4 + G7 (upper bound; G5/G6 not applied), whole eligible train pool")
share_table(c_all, c_zero, order)
for label, allowed in (("(a) table as proposed, all types", lambda t: True),
                       ("(b) information/other/support_groups removed", lambda t: t not in NO_SENTINEL)):
    n = sum(c_all[t] for t in order if allowed(t))
    z = sum(c_zero[t] for t in order if allowed(t))
    print(f"  {label}: pool-proportional zero-candidate share = {z}/{n} = {z / max(1, n):.1%}")

# ---------------------------------------------------------------- P3. can 500 be drawn; type mix vs hard negatives
ordered = sorted(pool, key=lambda r: hashlib.sha256(f"{SEED}:{r['record_id']}".encode()).hexdigest())
n_allowed = N_SENT - N_INFO_SENT
rest = [x for x in ordered[N_ANS:] if x["question_type"] not in NO_SENTINEL]
info_rest = [x for x in ordered[N_ANS:] if x["question_type"] == INFO]
s_allowed, s_info = rest[:n_allowed], info_rest[:N_INFO_SENT]
print(f"== P3. after the first {N_ANS} answerable draws (builder's seeded order): allowed-type records {len(rest)}"
      f" (need {n_allowed}: {len(rest) >= n_allowed}); information-type records {len(info_rest)}"
      f" (need {N_INFO_SENT}: {len(info_rest) >= N_INFO_SENT})")
mix = Counter(x["question_type"] for x in s_allowed)
z = sum(ncand[x["record_id"]] == 0 for x in s_allowed)
print(f"  {n_allowed} allowed-type sentinels: zero-candidate {z}/{n_allowed} = {z / n_allowed:.1%}")
print(f"  {N_INFO_SENT} information-type sentinels: 0 same-topic blocks by design")
print(f"  all {N_SENT} sentinels: 0 same-topic blocks (upper-bound availability) = {z + len(s_info)}/{N_SENT}"
      f" = {(z + len(s_info)) / N_SENT:.1%}")
dev_hn = Counter()
for line in open("artifacts/evaluation/evalsets/dev.jsonl", encoding="utf-8"):
    m = HN.fullmatch(json.loads(line).get("notes") or "")
    if m:
        dev_hn[m.group(1)] += 1
print("  question type: allowed-type sentinel draw | in code HARD_NEGATIVE_QTYPES | DEV hard negatives (missing type)")
for t in sorted(set(mix) | set(HARD_NEGATIVE_QTYPES) | set(dev_hn), key=lambda t: -mix[t]):
    print(f"    {t}: {mix[t]} ({mix[t] / n_allowed:.1%}) | {'yes' if t in HARD_NEGATIVE_QTYPES else 'no'} | {dev_hn[t]}")
hn_share = sum(mix[t] for t in HARD_NEGATIVE_QTYPES) / n_allowed
print(f"  share of the {n_allowed} allowed-type sentinels whose type is in HARD_NEGATIVE_QTYPES: {hn_share:.1%}")

# ---------------------------------------------------------------- P1. serving 0..4 split of different-type blocks
from medquad_qa.retrieval.factory import build_retriever  # noqa: E402

bundle = build_retriever()
ret = bundle.retriever
print("statuses:", [(s.name, s.ok, s.version) for s in bundle.statuses])
assert ret is not None, "serving retriever not built"
diff_dist = Counter()
other = Counter()
for x in val.values():
    hits, name, warnings = ret.retrieve_with_info(x["question"], TOPK)
    other[f"retriever name {name}"] += 1
    other["lexical_fallback warnings"] += any("lexical_fallback" in str(w) for w in warnings)
    if not any(h.record_id == x["record_id"] for h in hits):
        continue
    diff = 0
    for h in hits:
        if h.record_id == x["record_id"] or not on(topic_key(h.topic), x["tkey"]):
            continue
        y = val.get(h.record_id)
        if y is None:
            other["on-topic hit outside validation"] += 1
        elif y["duplicate_group_id"] != x["duplicate_group_id"] and y["question_type"] != x["question_type"]:
            diff += 1
    diff_dist[diff] += 1
n = sum(diff_dist.values())
print(f"== P1. serving, gold retrieved (n={n}): other on-topic DIFFERENT-type blocks per question")
for b in range(0, 5):
    print(f"    {b}: {diff_dist[b]} ({diff_dist[b] / max(1, n):.1%})")
for k in sorted(other):
    print(f"  {k}: {other[k]}")
