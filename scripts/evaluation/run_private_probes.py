"""D-040: run the pre-registered private safety-probe reserve against the frozen safety rules (after E4).

Usage: python scripts/evaluation/run_private_probes.py --probes <file> --expect-sha256 <hash> --out <json>
Refuses if the probe file's sha256 differs from the hash registered before any fix (D-040).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from medquad_qa.evaluation.evalset import sha256_file
from medquad_qa.rag.safety import SAFETY_RULES_VERSION, check_question


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True)
    ap.add_argument("--expect-sha256", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    actual = sha256_file(args.probes)
    if actual != args.expect_sha256:
        print(f"sha256 mismatch: {actual} != {args.expect_sha256}", file=sys.stderr)
        return 2
    probes = json.loads(Path(args.probes).read_text(encoding="utf-8"))
    personal = [(q, check_question(q)) for q in probes["personal"]]
    general = [(q, check_question(q)) for q in probes["general"]]
    out = {
        "probes_sha256": actual,
        "safety_rules_version": SAFETY_RULES_VERSION,
        "personal_refused": sum(d.refuse for _, d in personal),
        "personal_total": len(personal),
        "general_refused": sum(d.refuse for _, d in general),
        "general_total": len(general),
        "personal": [{"q": q, "refuse": d.refuse, "rule": d.rule_id} for q, d in personal],
        "general": [{"q": q, "refuse": d.refuse, "rule": d.rule_id} for q, d in general],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in list(out)[:6]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
