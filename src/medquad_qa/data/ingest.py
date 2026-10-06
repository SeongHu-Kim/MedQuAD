"""Raw CSV ingestion (stdlib csv only). The raw file is never modified."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

from medquad_qa.contracts import ContractViolationError


@dataclass(frozen=True)
class RawRow:
    row_index: int  # 0-based data row, header excluded
    question: str
    answer: str
    source: str
    focus_area: str


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path: Path, expected_columns: list[str]) -> list[RawRow]:
    """Read every data row exactly as stored (no stripping, no type coercion)."""
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        if header != expected_columns:
            raise ContractViolationError(f"unexpected CSV header {header!r}; expected {expected_columns!r}")
        rows: list[RawRow] = []
        for i, values in enumerate(reader):
            if len(values) != len(header):
                raise ContractViolationError(f"row {i} has {len(values)} fields, expected {len(header)}")
            rec = dict(zip(header, values, strict=True))
            rows.append(RawRow(i, rec["question"], rec["answer"], rec["source"], rec["focus_area"]))
    return rows
