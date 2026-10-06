"""Independent verification of the data-steward's corpus, split manifest and leakage report (E1 sign-off).

Re-derives everything from the raw CSV with evaluator-written code (no imports from medquad_qa.data):
  reconciliation, record_id formula, uniqueness, manifest/corpus/export consistency, group purity, and
  cross-split overlap on folded topic, normalized question, non-boilerplate exact answer, and answer near-dups
  (word 5-gram Jaccard, candidate pairs via shingles with df <= MAX_DF, independent of the steward's thresholds).
Output: a JSON report with IDs and counts only (no dataset text).

Usage: python scripts/evaluation/verify_splits.py [--out artifacts/evaluation/split_verification/report.json]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "medquad.csv"
CORPUS = ROOT / "data/processed/corpus.jsonl"
MANIFEST = ROOT / "data/manifests/split_manifest.jsonl"
META = ROOT / "data/manifests/split_manifest.meta.json"
EXCL = ROOT / "data/manifests/exclusions.jsonl"
LEAK = ROOT / "data/manifests/leakage_report.json"
EXPORTS = ROOT / "data/processed/exports"
SPLITS = ("train", "validation", "test")
SHINGLE = 5
MAX_DF = 25  # looser than the steward's df<=10 so the near-dup sweep is not a re-run of the same filter
NEAR_J = 0.8
RESIDUAL_J = 0.5
MIN_SHARED_RARE = 5
BOILERPLATE_MIN_TOPICS = 3


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fold(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def fold_topic_no_brackets(text: str) -> str:
    return fold(re.sub(r"\([^)]*\)", " ", text))


_FAMILY_SUFFIX = re.compile(r"(?:\s+(?:type|types|group|form|subtype))?(?:\s+(?:[0-9]+[a-z]?|[ivx]+|[a-z]))+$")


def topic_family(text: str) -> str:
    """Folded topic with trailing subtype markers removed: 'Noonan syndrome 2' -> 'noonan syndrome'."""
    return _FAMILY_SUFFIX.sub("", fold(text)).strip()


def ws(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def shingles(text: str) -> set[str]:
    toks = fold(text).split()
    return {" ".join(toks[i : i + SHINGLE]) for i in range(max(0, len(toks) - SHINGLE + 1))}


def read_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artifacts/evaluation/split_verification/report.json"))
    args = ap.parse_args()
    errors: list[str] = []
    rep: dict = {"inputs": {}}
    for p in (RAW, CORPUS, MANIFEST, META, EXCL, LEAK):
        rep["inputs"][str(p.relative_to(ROOT))] = sha256_file(p)

    # ---------------- raw
    with RAW.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    rep["raw_rows"] = len(rows)

    corpus = read_jsonl(CORPUS)
    manifest = read_jsonl(MANIFEST)
    excl = read_jsonl(EXCL)
    meta = json.loads(META.read_text())
    leak = json.loads(LEAK.read_text())
    rep["corpus_records"], rep["exclusions"] = len(corpus), len(excl)

    # ---------------- reconciliation: every raw row accounted for exactly once
    owner: dict[int, str] = {}
    dup_rows_claimed: set[int] = set()
    for r in corpus:
        ri = r["provenance"]["row_index"]
        if ri in owner:
            errors.append(f"row {ri} owned twice")
        owner[ri] = r["record_id"]
        dup_rows_claimed.update(r["provenance"]["duplicate_row_indices"])
    excl_rows = Counter(e["row_index"] for e in excl)
    for ri, c in excl_rows.items():
        if c > 1:
            errors.append(f"row {ri} excluded {c} times")
        if ri in owner:
            errors.append(f"row {ri} both a record and excluded")
    all_rows = set(owner) | set(excl_rows)
    if all_rows != set(range(len(rows))):
        errors.append(f"reconciliation: {len(set(range(len(rows))) - all_rows)} rows unaccounted")
    dup_excl = {e["row_index"] for e in excl if e["reason"] == "exact_duplicate_row"}
    if dup_excl != dup_rows_claimed:
        errors.append("exact_duplicate_row exclusions != union of Provenance.duplicate_row_indices")
    rep["reconciliation"] = f"{len(owner)} + {len(excl_rows)} = {len(owner) + len(excl_rows)} (raw {len(rows)})"
    rep["exclusions_by_reason"] = dict(Counter(e["reason"] for e in excl))

    # ---------------- record_id formula + raw text preserved + uniqueness
    bad_id = bad_raw = 0
    for r in corpus:
        raw = rows[r["provenance"]["row_index"]]
        payload = "\x1f".join(["medquad-rid-v1", raw["source"], raw["focus_area"], raw["question"], raw["answer"]])
        if r["record_id"] != "mq-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]:
            bad_id += 1
        if r["question_raw"] != raw["question"] or r["answer_raw"] != raw["answer"]:
            bad_raw += 1
    ids = [r["record_id"] for r in corpus]
    if bad_id:
        errors.append(f"record_id formula mismatch on {bad_id} records")
    if bad_raw:
        errors.append(f"raw text not preserved on {bad_raw} records")
    if len(set(ids)) != len(ids):
        errors.append("record_id not unique")
    rep["record_id_formula_mismatches"], rep["raw_text_mismatches"] = bad_id, bad_raw

    # ---------------- manifest vs corpus vs exports
    by_id = {r["record_id"]: r for r in corpus}
    split_of: dict[str, str] = {}
    for m in manifest:
        rid = m["record_id"]
        if rid in split_of:
            errors.append(f"manifest duplicate {rid}")
        split_of[rid] = m["split"]
        c = by_id.get(rid)
        if c is None:
            errors.append(f"manifest id {rid} not in corpus")
            continue
        for k in ("split_group_id", "duplicate_group_id", "topic", "question_type"):
            if m[k] != c[k]:
                errors.append(f"manifest/corpus mismatch {rid}.{k}")
    if set(split_of) != set(by_id):
        errors.append("manifest does not cover corpus exactly")
    for s in SPLITS:
        exp_ids = [r["record_id"] for r in read_jsonl(EXPORTS / f"records_{s}.jsonl")]
        want = {rid for rid, sp in split_of.items() if sp == s}
        if set(exp_ids) != want or len(exp_ids) != len(want):
            errors.append(f"export records_{s}.jsonl disagrees with manifest")
    rep["records_per_split"] = dict(Counter(split_of.values()))

    # ---------------- group purity
    def purity(key: str) -> int:
        sp: dict[str, set[str]] = defaultdict(set)
        for r in corpus:
            sp[r[key]].add(split_of[r["record_id"]])
        return sum(1 for v in sp.values() if len(v) > 1)

    rep["impure_split_groups"] = purity("split_group_id")
    rep["impure_duplicate_groups"] = purity("duplicate_group_id")
    if rep["impure_split_groups"] or rep["impure_duplicate_groups"]:
        errors.append("group spans splits")
    rep["groups_per_split"] = {
        s: len({r["split_group_id"] for r in corpus if split_of[r["record_id"]] == s}) for s in SPLITS
    }

    # ---------------- cross-split overlap on evaluator-defined keys
    def cross(keyfn, skip=lambda r: False) -> list[list[str]]:
        groups: dict[str, list[str]] = defaultdict(list)
        for r in corpus:
            if skip(r):
                continue
            k = keyfn(r)
            if k:
                groups[k].append(r["record_id"])
        return [sorted(v) for v in groups.values() if len({split_of[i] for i in v}) > 1]

    topics_per_answer: dict[str, set[str]] = defaultdict(set)
    for r in corpus:
        topics_per_answer[ws(r["answer"])].add(fold(r["topic"] or ""))
    boiler = {a for a, t in topics_per_answer.items() if len(t) >= BOILERPLATE_MIN_TOPICS}
    # "Boilerplate" shared only within one or two disease families (numbered subtypes) is disease-specific content.
    families_per_answer = {a: {topic_family(t) for t in ts} for a, ts in topics_per_answer.items()}
    family_content = {a for a in boiler if len(families_per_answer[a]) <= 2}
    overlap = {
        "folded_topic": cross(lambda r: fold(r["topic"] or "")),
        "normalized_question": cross(lambda r: fold(r["question"])),
        "non_boilerplate_exact_answer": cross(lambda r: ws(r["answer"]), skip=lambda r: ws(r["answer"]) in boiler),
        "family_content_exact_answer": cross(
            lambda r: ws(r["answer"]), skip=lambda r: ws(r["answer"]) not in family_content
        ),
    }
    diag = {
        "bracket_stripped_topic": cross(lambda r: fold_topic_no_brackets(r["topic"] or "")),
        "generic_boilerplate_exact_answer": cross(
            lambda r: ws(r["answer"]), skip=lambda r: ws(r["answer"]) not in boiler - family_content
        ),
        "topic_family": cross(lambda r: topic_family(r["topic"] or "")),
    }
    rep["boilerplate_answers_evaluator"] = len(boiler)
    rep["family_content_answers_evaluator"] = len(family_content)

    # near-dup answers
    sh = {r["record_id"]: shingles(r["answer"]) for r in corpus}
    df: dict[str, list[str]] = defaultdict(list)
    for rid, s in sh.items():
        for g in s:
            df[g].append(rid)
    cand: Counter[tuple[str, str]] = Counter()
    for ids_ in df.values():
        if 2 <= len(ids_) <= MAX_DF:
            for a, b in combinations(sorted(ids_), 2):
                if split_of[a] != split_of[b]:
                    cand[(a, b)] += 1
    near, template, residual = [], [], []
    for (a, b), _ in cand.items():
        sa, sb = sh[a], sh[b]
        j = len(sa & sb) / len(sa | sb)
        ra = {g for g in sa if len(df[g]) <= MAX_DF}
        rb = {g for g in sb if len(df[g]) <= MAX_DF}
        rare_j = len(ra & rb) / len(ra | rb) if ra | rb else 0.0
        row = [a, b, round(j, 4), round(rare_j, 4), len(ra & rb)]
        # content near-dup: overall J>=0.8 AND the rare (topic-specific) shingles also match; else shared template
        if j >= NEAR_J and rare_j >= NEAR_J and len(ra & rb) >= MIN_SHARED_RARE:
            near.append(row)
        elif j >= NEAR_J:
            template.append(row)
        elif j >= RESIDUAL_J:
            residual.append(row)
    overlap["answer_content_near_dup"] = sorted(near)
    diag["answer_template_near_dup"] = sorted(template)
    diag["answer_residual_0.5<=j<0.8"] = sorted(residual)
    rep["cross_split_overlap_counts"] = {k: len(v) for k, v in overlap.items()}
    rep["cross_split_overlap_examples"] = {k: v[:20] for k, v in overlap.items()}
    rep["diagnostic_counts"] = {k: len(v) for k, v in diag.items()}
    rep["diagnostic_examples"] = {k: v[:50] for k, v in diag.items()}
    for k, v in overlap.items():
        if v:
            errors.append(f"cross-split overlap on {k}: {len(v)}")

    # ---------------- agreement with steward's report
    rep["steward_report"] = {
        "passed": leak.get("passed"),
        "blocking_checks": leak.get("blocking_checks"),
        "split_version": leak.get("split_version"),
        "corpus_version": leak.get("corpus_version"),
        "diagnostics_counts": {k: v for k, v in leak.get("diagnostics", {}).items() if not isinstance(v, list)},
    }
    if leak.get("corpus_version") != meta.get("corpus_version"):
        errors.append("corpus_version differs between leakage report and split meta")
    rep["errors"] = errors
    rep["passed"] = not errors
    rep["parameters"] = {
        "shingle": SHINGLE,
        "max_df": MAX_DF,
        "near_j": NEAR_J,
        "residual_j": RESIDUAL_J,
        "min_shared_rare": MIN_SHARED_RARE,
        "boilerplate_min_topics": BOILERPLATE_MIN_TOPICS,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    keys = ("passed", "errors", "reconciliation", "cross_split_overlap_counts", "diagnostic_counts")
    print(json.dumps({k: rep[k] for k in (*keys, "records_per_split", "groups_per_split")}, indent=1))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
