"""Deterministic content-derived identifiers."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

SEP = "\x1f"
RECORD_ID_SALT = "medquad-rid-v1"


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def record_id(source: str, focus_area_raw: str, question_raw: str, answer_raw: str) -> str:
    """'mq-' + 16 hex of the raw row content. Row index is deliberately excluded."""
    payload = SEP.join([RECORD_ID_SALT, source, focus_area_raw, question_raw, answer_raw])
    return "mq-" + sha256_hex(payload)[:16]


def content_hash(question: str, answer: str) -> str:
    """sha256 of NORMALIZED question + US + NORMALIZED answer (contract definition)."""
    return sha256_hex(question + SEP + answer)


def group_id(prefix: str, member_record_ids: Iterable[str]) -> str:
    """Group ID from its sorted member record IDs, so it only changes when membership changes."""
    return f"{prefix}-" + sha256_hex(SEP.join(sorted(member_record_ids)))[:16]
