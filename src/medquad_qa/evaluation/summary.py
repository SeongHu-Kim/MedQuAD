"""Offline-evaluation summary for the API's ``medquad_offline_eval_metric`` gauges (agreed format).

{"eval_run_id": str, "completed_at": ISO-8601 with tz, "metrics": [{"track", "mode", "metric", "value"}]}
Label values match ``LABEL_RE``; values are finite floats; at most ``MAX_ROWS`` rows; None metrics are omitted.
Written atomically (temp file + rename) because the API reloads on mtime change.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

LABEL_RE = re.compile(r"^[A-Za-z0-9_.:+-]{1,64}$")
MAX_ROWS = 500


def metric_name(key: str) -> str:
    """'recall@5' -> 'recall_at_5'; other characters outside the label alphabet -> '_'."""
    return re.sub(r"[^A-Za-z0-9_.:+-]", "_", key.replace("@", "_at_"))


def build_summary(eval_run_id: str, rows: Iterable[tuple[str, str, str, float | int | None]]) -> dict[str, object]:
    metrics = []
    for track, mode, metric, value in rows:
        if value is None or isinstance(value, bool):
            continue
        v = float(value)
        if not math.isfinite(v):
            continue
        labels = (track, mode.replace(":", "_"), metric_name(metric))
        for lab in (eval_run_id, *labels):
            if not LABEL_RE.match(lab):
                raise ValueError(f"invalid label value {lab!r}")
        metrics.append({"track": labels[0], "mode": labels[1], "metric": labels[2], "value": v})
    if len(metrics) > MAX_ROWS:
        raise ValueError(f"{len(metrics)} rows exceed {MAX_ROWS}")
    return {
        "eval_run_id": eval_run_id,
        "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "metrics": metrics,
    }


def write_summary_atomic(summary: dict[str, object], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=1, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
