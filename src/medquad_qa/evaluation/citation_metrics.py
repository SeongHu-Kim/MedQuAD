"""Citation validity, coverage and support, and the unsupported-claim rate.

Per answered (non-abstained) response:
- validity = |valid citations| / (|valid| + |invalid_citation_ids|); None when the answer cites nothing.
- invariant_violations: cited record IDs (Citation objects or inline ``[mq-…]`` markers) not in
  ``retrieved_record_ids`` — must be 0 for a correct pipeline (QAResponse invariant).
- coverage = claims carrying >=1 inline citation / claims.
- support_rate = cited claims supported by at least one of THEIR cited records / cited claims.
- unsupported_claim_rate = claims not supported by ANY record in the evidence pool / claims.
  The pool is the supplied evidence for RAG modes; for cross-mode comparison the caller may pass the gold
  reference records instead (reported as ``reference_unsupported_claim_rate``).
"Supported" means scorer(evidence, claim) >= threshold; the scorer name and threshold are recorded.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from medquad_qa.contracts.qa import QAResponse
from medquad_qa.evaluation.claims import CITATION_MARKER, LABEL_MARKER, Claim, extract_claims
from medquad_qa.evaluation.support import SupportScorer


class CitationScores(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str
    n_claims: int
    n_cited_claims: int
    n_supported_cited_claims: int
    n_unsupported_claims: int
    n_valid_citations: int
    n_invalid_citations: int
    invariant_violations: list[str]
    leaked_labels: int  # unmapped [E#] markers left in the final answer text
    validity: float | None
    coverage: float | None
    support_rate: float | None
    unsupported_claim_rate: float | None
    scorer: str
    threshold: float


def _score_pairs(scorer: SupportScorer, pairs: list[tuple[str, str]]) -> list[float]:
    many = getattr(scorer, "score_many", None)
    if callable(many) and pairs:
        return list(many(pairs))
    return [scorer.score(e, c) for e, c in pairs]


def _supported(
    claims: Sequence[Claim],
    pools: Sequence[Sequence[str]],
    evidence: Mapping[str, str],
    scorer: SupportScorer,
    thr: float,
) -> list[bool]:
    """claims[i] is supported if any record in pools[i] (with known text) scores >= thr."""
    pairs: list[tuple[str, str]] = []
    owner: list[int] = []
    for i, (claim, pool) in enumerate(zip(claims, pools, strict=True)):
        for rid in pool:
            if rid in evidence:
                pairs.append((evidence[rid], claim.text))
                owner.append(i)
    result = [False] * len(claims)
    for i, s in zip(owner, _score_pairs(scorer, pairs), strict=True):
        if s >= thr:
            result[i] = True
    return result


def score_response(
    response: QAResponse,
    evidence: Mapping[str, str],
    scorer: SupportScorer,
    threshold: float = 0.5,
    unsupported_pool: Sequence[str] | None = None,
) -> CitationScores:
    """Score one non-abstained response. ``evidence`` maps record_id -> full evidence text.

    ``unsupported_pool`` defaults to ``response.retrieved_record_ids`` (the supplied evidence).
    """
    if response.abstained:
        raise ValueError(f"{response.request_id}: abstained responses have no claims to score")
    claims = extract_claims(response.answer)
    supplied = set(response.retrieved_record_ids)
    inline = set(CITATION_MARKER.findall(response.answer))
    cited = {c.record_id for c in response.citations} | inline
    violations = sorted(cited - supplied)
    n_valid, n_invalid = len(response.citations), len(response.invalid_citation_ids)

    cited_claims = [c for c in claims if c.cited_ids]
    sup_cited = _supported(cited_claims, [c.cited_ids for c in cited_claims], evidence, scorer, threshold)
    pool = list(unsupported_pool if unsupported_pool is not None else response.retrieved_record_ids)
    sup_any = _supported(claims, [pool] * len(claims), evidence, scorer, threshold)

    return CitationScores(
        request_id=response.request_id,
        n_claims=len(claims),
        n_cited_claims=len(cited_claims),
        n_supported_cited_claims=sum(sup_cited),
        n_unsupported_claims=len(claims) - sum(sup_any),
        n_valid_citations=n_valid,
        n_invalid_citations=n_invalid,
        invariant_violations=violations,
        leaked_labels=len(LABEL_MARKER.findall(response.answer)),
        validity=n_valid / (n_valid + n_invalid) if n_valid + n_invalid else None,
        coverage=len(cited_claims) / len(claims) if claims else None,
        support_rate=sum(sup_cited) / len(cited_claims) if cited_claims else None,
        unsupported_claim_rate=(len(claims) - sum(sup_any)) / len(claims) if claims else None,
        scorer=scorer.name,
        threshold=threshold,
    )


def aggregate(scores: Sequence[CitationScores]) -> dict[str, float | int | None]:
    """Micro-averages over claims/citations (each claim weighs equally), plus response-level counts."""

    def ratio(num: int, den: int) -> float | None:
        return num / den if den else None

    n_claims = sum(s.n_claims for s in scores)
    n_cited = sum(s.n_cited_claims for s in scores)
    valid = sum(s.n_valid_citations for s in scores)
    invalid = sum(s.n_invalid_citations for s in scores)
    supported_cited = sum(s.n_supported_cited_claims for s in scores)
    unsupported = sum(s.n_unsupported_claims for s in scores)
    return {
        "n_responses": len(scores),
        "n_claims": n_claims,
        "citation_validity": ratio(valid, valid + invalid),
        "citation_coverage": ratio(n_cited, n_claims),
        "citation_support": ratio(supported_cited, n_cited),
        "unsupported_claim_rate": ratio(unsupported, n_claims),
        "responses_with_invariant_violation": sum(1 for s in scores if s.invariant_violations),
        "responses_with_leaked_labels": sum(1 for s in scores if s.leaked_labels),
        "responses_without_citation": sum(1 for s in scores if s.n_valid_citations == 0),
    }
