"""Export the latest offline evaluation summary as ``medquad_offline_eval_*`` gauges.

Owner: service-platform-engineer. The evaluator writes the summary; this module only reads it.
Expected JSON (see docs/service/metrics.md)::

    {"eval_run_id": "e4-...", "completed_at": "2026-10-07T12:00:00+00:00",
     "metrics": [{"track": "B", "mode": "rag", "metric": "citation_validity", "value": 0.97}, ...]}

The file is re-read when its mtime changes (checked at scrape time), so a new offline run shows up
without restarting the API. A missing or malformed file clears the gauges and never fails a scrape.
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime
from pathlib import Path

from medquad_qa.observability.metrics import ServiceMetrics

log = logging.getLogger(__name__)

_LABEL_RE = re.compile(r"^[A-Za-z0-9_.:+-]{1,64}$")
MAX_SERIES = 500


class OfflineEvalExporter:
    def __init__(self, metrics: ServiceMetrics, path: str | None) -> None:
        self._m = metrics
        self._path = Path(path) if path else None
        self._mtime: float | None = None
        self.loaded_run_id: str | None = None

    def refresh(self) -> None:
        if self._path is None:
            return
        try:
            mtime = self._path.stat().st_mtime
        except OSError:
            if self._mtime is not None:
                self._clear()
                self._mtime = None
            return
        if mtime == self._mtime:
            return
        self._mtime = mtime
        self._clear()
        try:
            self._load()
        except (OSError, ValueError, TypeError, KeyError) as exc:
            log.warning("offline eval summary ignored", extra={"path": str(self._path), "error": type(exc).__name__})
            self._clear()

    def _clear(self) -> None:
        self._m.offline_eval.clear()
        self._m.offline_eval_timestamp.clear()
        self.loaded_run_id = None

    def _load(self) -> None:
        assert self._path is not None
        data = json.loads(self._path.read_text(encoding="utf-8"))
        run_id = str(data["eval_run_id"])
        if not _LABEL_RE.match(run_id):
            raise ValueError("eval_run_id has unsupported characters")
        rows = data["metrics"]
        if not isinstance(rows, list) or len(rows) > MAX_SERIES:
            raise ValueError(f"metrics must be a list of at most {MAX_SERIES} rows")
        for row in rows:
            labels = {k: str(row[k]) for k in ("track", "mode", "metric")}
            if not all(_LABEL_RE.match(v) for v in labels.values()):
                raise ValueError("label value has unsupported characters")
            value = float(row["value"])
            if not math.isfinite(value):
                raise ValueError("non-finite metric value")
            self._m.offline_eval.labels(eval_run_id=run_id, **labels).set(value)
        completed = data.get("completed_at")
        if completed:
            ts = datetime.fromisoformat(str(completed)).timestamp()
            self._m.offline_eval_timestamp.labels(eval_run_id=run_id).set(ts)
        self.loaded_run_id = run_id
