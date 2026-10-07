"""Per-mode summary of QAResponses for one frozen query set (no text scoring; see citation_metrics/rubric).

Covers: abstention confusion, warning rates (incl. D-022 ``evidence_truncated``), matched-budget evidence
(effective generation params must be identical across a run), finish_reason=length rate, latency, and
version fingerprints. Raises if responses do not align 1:1 with examples or mix modes/versions.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from medquad_qa.contracts.evaluation import EvaluationExample
from medquad_qa.contracts.qa import RAG_MODES, QAResponse
from medquad_qa.evaluation import abstention, latency
from medquad_qa.evaluation.retrieval_metrics import aggregate as aggregate_retrieval
from medquad_qa.evaluation.retrieval_metrics import score_query

RESOURCE_LIST_PREFIX = "These resources address the diagnosis or management of"


def is_resource_list_gold(example: EvaluationExample, answers: dict[str, str]) -> bool:
    """Pre-registered secondary-analysis subgroup (2026-10-06, before any TEST run): every gold record's answer
    is a GHR resource-link list (starts with ``RESOURCE_LIST_PREFIX``), so abstaining is arguably grounded.

    Frozen labels are NOT changed; this subgroup is reported separately and excluded only in secondary tables.
    """
    return bool(example.gold_record_ids) and all(
        answers.get(g, "").startswith(RESOURCE_LIST_PREFIX) for g in example.gold_record_ids
    )


def expected_for_mode(example: EvaluationExample, mode: str) -> str:
    """D-023: personalized-advice refusals apply in all modes; evidence-based abstentions only in RAG modes.

    Labels in the frozen set describe RAG behaviour. In closed-book modes an evidence-based ``abstain``
    expectation becomes ``either`` (no retrieval, so no evidence check exists to trigger it).
    """
    if mode not in RAG_MODES and example.expected_behavior == "abstain" and example.case_type != "personalized_advice":
        return "either"
    return example.expected_behavior


def abstention_group(example: EvaluationExample) -> str:
    tag = (example.notes or "").split(":", 1)[0] if example.notes else ""
    return f"{example.case_type}:{tag}" if tag else example.case_type


def summarize_mode(
    examples: Sequence[EvaluationExample],
    responses: Sequence[QAResponse],
    *,
    retrieval_ks: Sequence[int] = (1, 3, 5),
) -> dict[str, Any]:
    if len(examples) != len(responses):
        raise ValueError("examples and responses must align 1:1")
    modes = {r.experiment_mode for r in responses}
    if len(modes) != 1:
        raise ValueError(f"mixed experiment modes in one run: {sorted(modes)}")
    mode = modes.pop()
    # Fixture-served items (Track C conflict/injection; retriever "fixture") legitimately report the fixture corpus.
    real = [r for r in responses if r.retriever != "fixture"] or list(responses)
    versions = {
        k: sorted({str(getattr(r, k)) for r in (responses if k in ("model_version", "prompt_version") else real)})
        for k in ("model_version", "prompt_version", "corpus_version", "index_version", "retriever")
    }
    versions["n_fixture_served"] = [str(sum(r.retriever == "fixture" for r in responses))]
    for k, v in versions.items():
        if len(v) > 1 and k not in ("retriever", "n_fixture_served"):
            raise ValueError(f"{k} changed within one run: {v}")

    outcomes = [
        abstention.AbstentionOutcome(
            example_id=e.example_id,
            expected_behavior=expected_for_mode(e, mode),  # type: ignore[arg-type]
            abstained=r.abstained,
            expected_reason=e.expected_abstention_reason,
            actual_reason=r.abstention_reason,
            group=abstention_group(e),
        )
        for e, r in zip(examples, responses, strict=True)
    ]
    n = len(responses)
    warn = Counter(w for r in responses for w in set(r.warnings))
    gens = [r.generation for r in responses if r.generation is not None]
    params = {g.params.model_dump_json() for g in gens}

    out: dict[str, Any] = {
        "experiment_mode": mode,
        "n": n,
        "versions": versions,
        "abstention": abstention.confusion_by_group(outcomes),
        "abstention_reasons": abstention.reason_counts(outcomes),
        "warning_rates": {w: c / n for w, c in sorted(warn.items())},
        "evidence_truncated_rate": warn.get("evidence_truncated", 0) / n,
        "lexical_fallback_rate": warn.get("lexical_fallback", 0) / n,
        "generation": {
            "n_generated": len(gens),
            "distinct_effective_params": sorted(params),
            "matched_params": len(params) <= 1,
            "finish_length_rate": sum(g.finish_reason == "length" for g in gens) / len(gens) if gens else None,
            "completion_tokens_mean": sum(g.completion_tokens for g in gens) / len(gens) if gens else None,
            "prompt_tokens_max": max((g.prompt_tokens for g in gens), default=None),
        },
        "invalid_citation_responses": sum(bool(r.invalid_citation_ids) for r in responses),
        "latency": latency.summarize([r.latency_ms for r in responses]),
        "component_latency": latency.summarize_components([r.component_latency_ms for r in responses]),
    }
    if mode in RAG_MODES:
        scored = [
            score_query(e.example_id, r.retrieved_record_ids, set(e.gold_record_ids), ks=retrieval_ks)
            for e, r in zip(examples, responses, strict=True)
            if e.gold_record_ids
        ]
        out["supplied_evidence_retrieval"] = aggregate_retrieval(scored) if scored else None
    return out
