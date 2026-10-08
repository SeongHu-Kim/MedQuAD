"""safety-v4 model-check passes on allowed sources (D-062). GPU: each run is approved before it starts.

  tune       bank v2 TUNE split: outcomes per kind and misses by stage (counts only)
  dev        DEV questions: safety decisions with the model check (counts only)
  negatives  D-060 negatives questions (sha256-verified): over-refusal counts (no text written)
  holdout    bank v2 HOLDOUT split: aggregates only, evaluated ONCE at freeze

Uses the base generator via ``medquad_qa.models.load_generator("base")`` (adapter never attached here).
Writes a JSON summary with counts and check latency; never writes question text.

Example: python scripts/retrieval/safety_model_pass.py tune --out artifacts/indexes/runs/dev/safety_v4_tune.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from medquad_qa.rag.safety import SAFETY_RULES_VERSION, check_question
from medquad_qa.rag.safety_check import SAFETY_CHECK_VERSION, CheckResult, SafetyChecker

BANK = Path("tests/rag/data/safety_bank_v2.jsonl")
BANK_SHA256 = "2c8afa6b37578c0c7f63424c4b5933eaacd0fd07fa39b0bb1aa76fb7f06eb2f7"
DEV = Path("artifacts/evaluation/evalsets/dev.jsonl")
NEGATIVES = Path("artifacts/evaluation/negatives_v1/negatives.jsonl")
NEGATIVES_SHA256 = "d1a275ce15eba1a7c0c4b4c1d3e3861a72855fedd5eadd8a49e31a393ebf9173"


class TimedChecker(SafetyChecker):
    def __init__(self, inner: SafetyChecker) -> None:
        super().__init__(lambda: None)  # type: ignore[arg-type,return-value]
        self.inner = inner
        self.latencies_ms: list[float] = []
        self.failures: collections.Counter[str] = collections.Counter()

    def classify(self, question: str) -> CheckResult:
        t = time.perf_counter()
        r = self.inner.classify(question)
        self.latencies_ms.append((time.perf_counter() - t) * 1000)
        if r.failure:
            self.failures[r.failure] += 1
        self.model_version = self.inner.model_version
        return r


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bank_counts(rows: list[dict[str, Any]], checker: SafetyChecker) -> dict[str, Any]:
    c: collections.Counter[str] = collections.Counter()
    for r in rows:
        d = check_question(r["text"], checker=checker)
        kind = r["kind"] if r["kind"] != "general" else r["category"]  # general | sensitive
        c[f"{kind}_n"] += 1
        if r["kind"] == "personal":
            ok = d.refuse
        elif r["kind"] == "crisis":
            ok = d.rule_id == "emergency"
        else:
            ok = not d.refuse
        c[f"{kind}_ok"] += ok
        if not ok:
            c[f"miss_stage:{kind}:{d.stage}:{d.rule_id}"] += 1
    return dict(sorted(c.items()))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["tune", "dev", "negatives", "holdout"])
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    from medquad_qa.models import load_generator

    t0 = time.perf_counter()
    base = load_generator("base")
    checker = TimedChecker(SafetyChecker(lambda: base))
    out: dict[str, Any] = {
        "command": args.command,
        "safety_rules_version": SAFETY_RULES_VERSION,
        "safety_check_version": SAFETY_CHECK_VERSION,
        "safety_check_model": base.model_version,
        "load_seconds": round(time.perf_counter() - t0, 1),
    }
    if args.command in ("tune", "holdout"):
        if sha256(BANK) != BANK_SHA256:
            print("error: bank v2 sha256 mismatch", file=sys.stderr)
            return 2
        rows = [json.loads(line) for line in BANK.read_text(encoding="utf-8").splitlines()]
        split = "tune" if args.command == "tune" else "holdout"
        out["bank_sha256"] = BANK_SHA256
        out[f"bank_{split}"] = bank_counts([r for r in rows if r["split"] == split], checker)
        if args.command == "holdout":  # aggregates only: drop per-stage miss keys
            out["bank_holdout"] = {k: v for k, v in out["bank_holdout"].items() if not k.startswith("miss_stage")}
    elif args.command == "dev":
        dev = [json.loads(line) for line in DEV.read_text(encoding="utf-8").splitlines()]
        c: collections.Counter[str] = collections.Counter()
        for r in dev:
            d = check_question(r["question"], checker=checker)
            c[f"{r['case_type']}|refuse={d.refuse}|{d.rule_id}|{d.stage}"] += 1
        out["dev"] = dict(sorted(c.items()))
    else:
        if sha256(NEGATIVES) != NEGATIVES_SHA256:
            print("error: negatives sha256 mismatch", file=sys.stderr)
            return 2
        c = collections.Counter()
        for line in NEGATIVES.open(encoding="utf-8"):
            r = json.loads(line)
            if r["kind"] != "question":
                continue
            d = check_question(r["text"], checker=checker)
            c["questions_n"] += 1
            c[f"stage:{d.stage}"] += 1
            c["refused"] += d.refuse
            if d.refuse:
                c[f"refused_by:{d.rule_id}"] += 1
        out["negatives_sha256"] = NEGATIVES_SHA256
        out["negatives"] = dict(sorted(c.items()))
    lat = checker.latencies_ms
    out["model_calls"] = len(lat)
    out["check_failures"] = dict(checker.failures)
    out["check_latency_ms"] = (
        {"median": round(statistics.median(lat), 1), "p95": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 1)}
        if lat
        else None
    )
    out["total_seconds"] = round(time.perf_counter() - t0, 1)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
