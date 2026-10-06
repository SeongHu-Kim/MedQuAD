"""Lead-owned repository hygiene checks (offline)."""

import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_invisible_or_control_chars_in_source() -> None:
    """Bidi/format/control chars must be written as escapes, never literally (e.g. in sanitiser regexes)."""
    offenders = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            if any(unicodedata.category(c) in ("Cf", "Cc") and c != "\t" for c in line):
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}")
    assert not offenders, offenders
