"""Citation validation and rewriting (D-010).

The model cites per-request labels ``[E1]..[Ek]``. Valid labels are rewritten to inline ``[mq-...]`` record
markers. A literal ``[mq-...]`` is accepted only if that record was supplied. Every other label/ID is
stripped from the text and reported in ``invalid_ids``. Invariant: cited ⊆ supplied ⊆ retrieved.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from medquad_qa.contracts import Citation, RetrievalHit
from medquad_qa.rag.prompts import SENTINEL

# One bracket group holding labels/IDs: [E1], [E1, E2], [ e3 ], [mq-0123456789abcdef], [E1; mq-...]
_GROUP_RE = re.compile(
    r"\[\s*((?:E\s*\d+|mq-[0-9A-Za-z]+)(?:\s*[,;]\s*(?:E\s*\d+|mq-[0-9A-Za-z]+))*)\s*\]", re.IGNORECASE
)
_TOKEN_RE = re.compile(r"E\s*\d+|mq-[0-9A-Za-z]+", re.IGNORECASE)
_VALID_RID = re.compile(r"^mq-[0-9a-f]{16}$")
_SENTINEL_RE = re.compile(r"\bINSUFFICIENT[_ ]EVIDENCE\b", re.IGNORECASE)
SNIPPET_CHARS = 300


@dataclass
class CitationResult:
    text: str
    cited_record_ids: list[str] = field(default_factory=list)  # first-appearance order, unique
    invalid_ids: list[str] = field(default_factory=list)  # raw tokens, unique, in order
    has_sentinel: bool = False


def label_map(supplied: Sequence[RetrievalHit]) -> dict[str, str]:
    """'E1' -> record_id for the supplied hits (labels follow supply order)."""
    return {f"E{i}": h.record_id for i, h in enumerate(supplied, start=1)}


def validate_citations(text: str, supplied: Sequence[RetrievalHit]) -> CitationResult:
    labels = label_map(supplied)
    supplied_ids = set(labels.values())
    cited: list[str] = []
    invalid: list[str] = []

    def _resolve(token: str) -> str | None:
        norm = re.sub(r"\s+", "", token)
        if norm[:1] in "eE" and norm[1:].isdigit():
            return labels.get("E" + str(int(norm[1:])))
        low = norm.lower()
        if _VALID_RID.match(low) and low in supplied_ids:
            return low
        return None

    def _replace(m: re.Match[str]) -> str:
        out: list[str] = []
        for tok in _TOKEN_RE.findall(m.group(1)):
            rid = _resolve(tok)
            if rid is None:
                raw = re.sub(r"\s+", "", tok)
                if raw not in invalid:
                    invalid.append(raw)
                continue
            if rid not in cited:
                cited.append(rid)
            marker = f"[{rid}]"
            if marker not in out:
                out.append(marker)
        return "".join(out)

    rewritten = _GROUP_RE.sub(_replace, text)
    # tidy whitespace left by stripped markers (" ." -> ".")
    rewritten = re.sub(r"[ \t]+([.,;:!?])", r"\1", rewritten)
    rewritten = re.sub(r"[ \t]{2,}", " ", rewritten).strip()
    return CitationResult(
        text=rewritten,
        cited_record_ids=cited,
        invalid_ids=invalid,
        has_sentinel=bool(_SENTINEL_RE.search(text)) or text.strip() == SENTINEL,
    )


def build_citations(cited_ids: Sequence[str], supplied: Sequence[RetrievalHit]) -> list[Citation]:
    by_id = {h.record_id: h for h in supplied}
    out = []
    for rid in cited_ids:
        h = by_id[rid]
        snippet = h.evidence_text if len(h.evidence_text) <= SNIPPET_CHARS else h.evidence_text[:SNIPPET_CHARS] + "…"
        out.append(
            Citation(
                record_id=h.record_id,
                chunk_id=h.chunk_id,
                source_name=h.source_name,
                source_url=h.source_url,
                topic=h.topic,
                evidence_snippet=snippet,
            )
        )
    return out
