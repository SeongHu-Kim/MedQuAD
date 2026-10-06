"""DEV-only proxy for citation coverage: share of answer sentences carrying >=1 [mq-...] marker.

Usage: python scripts/retrieval/citation_coverage_dev.py artifacts/indexes/runs/dev/pipeline_*.jsonl
Official citation metrics (incl. NLI support) are evaluation-safety-engineer's; this is a quick tuning signal.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\-*•])|\n+")
_MARK = re.compile(r"\[mq-[0-9a-f]{16}\]")


def coverage(path: Path) -> tuple[int, int, int]:
    answered = sentences = cited = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("abstained", True):
            continue
        answered += 1
        for sent in (s.strip() for s in _SENT.split(row["answer"])):
            if len(re.sub(r"\[mq-[0-9a-f]{16}\]", "", sent).strip(" -*•:")) < 3:
                continue
            sentences += 1
            cited += bool(_MARK.search(sent))
    return answered, sentences, cited


if __name__ == "__main__":
    for p in sys.argv[1:]:
        a, s, c = coverage(Path(p))
        print(f"{Path(p).name}: answered={a} sentences={s} cited={c} coverage={c / s if s else 0:.3f}")
