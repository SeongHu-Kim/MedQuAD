"""CPU-only F-009/F-010 remediation checks on allowed sources (D-060). Counts only; no derived text is written.

Usage:
  python scripts/retrieval/remediation_checks.py --out artifacts/indexes/runs/dev/remediation_checks.json [--holdout]

Sources: owner bank (tests/rag/data/safety_bank_v1.jsonl; tune split, plus holdout aggregates with --holdout),
DEV (artifacts/evaluation/evalsets/dev.jsonl), the D-060 negatives file (sha256-verified), the owner injection
bank (tests/rag/data/injection_bank_v1.jsonl) and the top-5 dense:qa evidence for every DEV question (CPU).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path

from medquad_qa.rag.evidence_filter import EVIDENCE_FILTER_VERSION, filter_evidence, is_injection
from medquad_qa.rag.prompts import CLOSED_BOOK_PROMPT_VERSION, PROMPT_VERSION
from medquad_qa.rag.safety import SAFETY_RULES_VERSION, check_question

NEGATIVES = Path("artifacts/evaluation/negatives_v1/negatives.jsonl")
NEGATIVES_SHA256 = "d1a275ce15eba1a7c0c4b4c1d3e3861a72855fedd5eadd8a49e31a393ebf9173"
BANK = Path("tests/rag/data/safety_bank_v1.jsonl")
INJ = Path("tests/rag/data/injection_bank_v1.jsonl")
DEV = Path("artifacts/evaluation/evalsets/dev.jsonl")
# Holdout items whose wording was later reused in owner tests during development (disclosed; excluded from the
# holdout go/no-go and reported separately).
CONTAMINATED_HOLDOUT = frozenset({"c-self_passive-01", "s-sensitive-040"})


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bank_counts(rows: list[dict]) -> dict[str, int]:
    c: collections.Counter[str] = collections.Counter()
    for r in rows:
        d = check_question(r["text"])
        if r["kind"] == "personal":
            c["personal_n"] += 1
            c["personal_refused"] += d.refuse
        elif r["kind"] == "crisis":
            c["crisis_n"] += 1
            c["crisis_emergency"] += d.rule_id == "emergency"
        else:
            c[f"{r['category']}_n"] += 1
            c[f"{r['category']}_refused"] += d.refuse
    return dict(c)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--holdout", action="store_true", help="also report holdout aggregates (owner go/no-go)")
    p.add_argument("--skip-dev-evidence", action="store_true")
    args = p.parse_args(argv)

    out: dict[str, object] = {
        "versions": {
            "safety_rules_version": SAFETY_RULES_VERSION,
            "prompt_version": PROMPT_VERSION,
            "closed_book_prompt_version": CLOSED_BOOK_PROMPT_VERSION,
            "evidence_filter_version": EVIDENCE_FILTER_VERSION,
        },
        "inputs_sha256": {"bank": sha256(BANK), "injection_bank": sha256(INJ), "dev": sha256(DEV)},
    }
    bank = [json.loads(line) for line in BANK.read_text(encoding="utf-8").splitlines()]
    out["bank_tune"] = bank_counts([r for r in bank if r["split"] == "tune"])
    if args.holdout:
        hold = [r for r in bank if r["split"] == "holdout"]
        out["bank_holdout_clean"] = bank_counts([r for r in hold if r["id"] not in CONTAMINATED_HOLDOUT])
        out["bank_holdout_all"] = bank_counts(hold)
        out["bank_holdout_contaminated_ids"] = sorted(CONTAMINATED_HOLDOUT)

    dev = [json.loads(line) for line in DEV.read_text(encoding="utf-8").splitlines()]
    out["dev_safety"] = {
        "personal_refused": sum(
            check_question(r["question"]).refuse for r in dev if r["case_type"] == "personalized_advice"
        ),
        "personal_n": sum(r["case_type"] == "personalized_advice" for r in dev),
        "others_refused": sum(
            check_question(r["question"]).refuse for r in dev if r["case_type"] != "personalized_advice"
        ),
        "others_n": sum(r["case_type"] != "personalized_advice" for r in dev),
    }

    if sha256(NEGATIVES) != NEGATIVES_SHA256:
        print("error: negatives sha256 mismatch", file=sys.stderr)
        return 2
    q_n = q_ref = s_n = s_flag = 0
    for line in NEGATIVES.open(encoding="utf-8"):
        r = json.loads(line)
        if r["kind"] == "question":
            q_n += 1
            q_ref += check_question(r["text"]).refuse
        elif r["kind"] == "answer_sentence":
            s_n += 1
            s_flag += is_injection(r["text"])
    out["negatives"] = {
        "sha256": NEGATIVES_SHA256,
        "questions_refused": q_ref,
        "questions_n": q_n,
        "answer_sentences_flagged": s_flag,
        "answer_sentences_n": s_n,
    }

    inj = [json.loads(line) for line in INJ.read_text(encoding="utf-8").splitlines()]
    by_cat: dict[str, dict[str, int]] = collections.defaultdict(lambda: {"n": 0, "payload_dropped_fact_kept": 0})
    for r in inj:
        fr = filter_evidence(r["evidence"])
        by_cat[r["category"]]["n"] += 1
        by_cat[r["category"]]["payload_dropped_fact_kept"] += r["canary"] not in fr.text and fr.text == r["fact"]
    out["injection_bank_filter"] = dict(by_cat)

    if not args.skip_dev_evidence:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from medquad_qa.retrieval.factory import build_retriever, require_retriever
        from medquad_qa.retrieval.settings import RetrievalSettings

        retr = require_retriever(
            build_retriever(
                RetrievalSettings.from_env(retriever="dense_fallback", index_text_mode="question_answer", device="cpu")
            )
        )
        n = flagged = 0
        for item in dev:
            for h in retr.retrieve(item["question"], 5):
                n += 1
                flagged += filter_evidence(h.evidence_text).dropped > 0
        out["dev_top5_evidence"] = {"chunks": n, "chunks_with_dropped_sentence": flagged, "retriever": retr.name}

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
