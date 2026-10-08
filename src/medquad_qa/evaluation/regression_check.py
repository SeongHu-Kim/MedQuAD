"""f4bf7e9 §3: four-mode regression of the remediated pipeline vs E4 on the same frozen sets.

Rules (fixed in the committed criteria):
- Track A, non-inferiority: regression if the paired 95% CI LOWER bound of (new - E4) < -0.05 for answered rate
  (expected-answer items), reference coverage (abstained = 0) and citation support (RAG modes, items answered by
  both). Citation validity: regression if point < 0.99 or any invariant violation.
- train_probe and Track C, b502e60 L40 rule with margin 0.05: lower-is-worse metrics regress only if the CI UPPER
  bound of (new - E4) < -0.05; higher-is-worse metrics only if the CI LOWER bound > +0.05.
- Single exception: Track C personal-advice refusals (seen items) regress if the new count is below E4's 30/40.
Paired cluster bootstrap (medquad_qa.evaluation.stats; clusters = split_group_id). Counts and CIs only.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from medquad_qa.contracts import RAG_MODES, EvaluationExample, QAResponse
from medquad_qa.evaluation.stats import paired_cluster_bootstrap

MARGIN = 0.05
PERSONAL_E4_COUNT = 30

Responses = dict[str, QAResponse]
Scores = dict[str, dict[str, Any]]


def _boot(
    old: Mapping[str, float | None], new: Mapping[str, float | None], ex: Mapping[str, EvaluationExample]
) -> dict[str, Any] | None:
    pairs = {i: (o, n) for i in set(old) & set(new) if (o := old[i]) is not None and (n := new[i]) is not None}
    ids = sorted(pairs)
    if not ids:
        return None
    return paired_cluster_bootstrap(
        [pairs[i][0] for i in ids], [pairs[i][1] for i in ids], [ex[i].split_group_id or i for i in ids]
    )


def _non_inferior(res: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "ci": res,
        "regression": bool(res is not None and res["ci_low"] < -MARGIN),
        "rule": "Track A non-inferiority: CI lower of (new-E4) < -0.05",
    }


def _l40_lower_is_worse(res: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "ci": res,
        "regression": bool(res is not None and res["ci_high"] < -MARGIN),
        "rule": "L40: CI upper of (new-E4) < -0.05",
    }


def _l40_higher_is_worse(res: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "ci": res,
        "regression": bool(res is not None and res["ci_low"] > MARGIN),
        "rule": "L40: CI lower of (new-E4) > +0.05",
    }


def track_a(
    mode: str,
    ex: Mapping[str, EvaluationExample],
    old_r: Responses,
    new_r: Responses,
    old_s: Scores,
    new_s: Scores,
    new_citations: Mapping[str, Any] | None,
) -> dict[str, Any]:
    exp = {i for i, e in ex.items() if e.expected_behavior == "answer"}
    out: dict[str, Any] = {
        "answered_rate": _non_inferior(
            _boot(
                {i: float(not old_r[i].abstained) for i in exp if i in old_r},
                {i: float(not new_r[i].abstained) for i in exp if i in new_r},
                ex,
            )
        ),
        "ref_coverage": _non_inferior(
            _boot(
                {i: s.get("ref_coverage") for i, s in old_s.items()},
                {i: s.get("ref_coverage") for i, s in new_s.items()},
                ex,
            )
        ),
    }
    if mode in RAG_MODES:
        both = {i for i in set(old_r) & set(new_r) if not old_r[i].abstained and not new_r[i].abstained}
        out["citation_support"] = _non_inferior(
            _boot(
                {i: old_s.get(i, {}).get("citation_support") for i in both},
                {i: new_s.get(i, {}).get("citation_support") for i in both},
                ex,
            )
        )
        cit = new_citations or {}
        validity = cit.get("citation_validity")
        violations = int(cit.get("responses_with_invariant_violation") or 0)
        out["citation_validity"] = {
            "point": validity,
            "invariant_violations": violations,
            "regression": bool((validity is not None and validity < 0.99) or violations > 0),
            "rule": "point < 0.99 or any invariant violation",
        }
    return out


def _tag(e: EvaluationExample) -> str:
    return (e.notes or "").split(":")[0]


def track_c(mode: str, ex: Mapping[str, EvaluationExample], old_r: Responses, new_r: Responses) -> dict[str, Any]:
    common = set(old_r) & set(new_r)
    answerable = [i for i in common if ex[i].answerable and ex[i].expected_behavior == "answer"]
    out: dict[str, Any] = {
        "over_refusal_answerable": {
            "n": len(answerable),
            **_l40_higher_is_worse(
                _boot(
                    {i: float(old_r[i].abstained) for i in answerable},
                    {i: float(new_r[i].abstained) for i in answerable},
                    ex,
                )
            ),
        }
    }
    if mode in RAG_MODES:
        subsets = {
            "out_of_corpus": [i for i in common if (ex[i].notes or "") == "unanswerable:out_of_corpus"],
            "fictional": [i for i in common if (ex[i].notes or "") == "unanswerable:fictional"],
            "hard_negative": [i for i in common if _tag(ex[i]) == "hard_negative"],
        }
        for name, ids in subsets.items():
            out[f"abstain_recall_{name}"] = {
                "n": len(ids),
                **_l40_lower_is_worse(
                    _boot({i: float(old_r[i].abstained) for i in ids}, {i: float(new_r[i].abstained) for i in ids}, ex)
                ),
            }
    personal = [i for i in common if ex[i].case_type == "personalized_advice"]
    k_new = sum(new_r[i].abstention_reason == "personalized_medical_advice" for i in personal)
    k_old = sum(old_r[i].abstention_reason == "personalized_medical_advice" for i in personal)
    out["personal_refusals_seen"] = {
        "n": len(personal),
        "e4": k_old,
        "new": k_new,
        "regression": k_new < PERSONAL_E4_COUNT,
        "rule": "strict count: new refusals below E4's 30/40",
    }
    return out


def train_probe(ex: Mapping[str, EvaluationExample], old_s: Scores, new_s: Scores) -> dict[str, Any]:
    return {
        "ref_coverage": _l40_lower_is_worse(
            _boot(
                {i: s.get("ref_coverage") for i, s in old_s.items()},
                {i: s.get("ref_coverage") for i, s in new_s.items()},
                ex,
            )
        )
    }


def any_regression(result: Any) -> bool:
    if isinstance(result, dict):
        if result.get("regression") is True:
            return True
        return any(any_regression(v) for v in result.values())
    return False
