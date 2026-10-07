"""E4b comparisons: adapter v2c vs the E4 runs (rag, finetuned v1, finetuned_rag v1), as pre-declared in b502e60 and
docs/evaluation/e4b_predeclaration_amendment_1.md.

Inputs are the stored per-item QAResponses and the per-item scores written by ``scripts/evaluation/score_e4.py`` for
BOTH trees. score_e4.py itself is not changed: E4 and E4b are scored by the same bytes, and ``check_provenance``
refuses to compare if the two metrics.json provenance blocks disagree.

Primary family (Holm over exactly these 10 tests):
  P1 Track A, finetuned_rag v2c vs rag: (a) answered [McNemar]; (b) citation validity, (c) citation support,
     (d) unsupported-claim rate vs supplied evidence [bootstrap on items answered by both]; (e) reference coverage,
     abstained = 0 [bootstrap, all items].
  P2 Track A, finetuned_rag v2c vs v1: answered [McNemar].
  P3 Track C, finetuned_rag v2c vs rag: over-refusal on answerable items (general controls + adversarial, i.e. all
     answerable=True items) [McNemar on abstained]; abstention recall on out-of-corpus, fictional and hard-negative
     items, each separately [McNemar on abstained].
Not in the family: injection canary leaks (count), conflicting-evidence outcomes (descriptive), the regression check,
the train_probe secondary analysis, risk metrics and the amendment §E descriptive analyses.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from medquad_qa.contracts import EvaluationExample, QAResponse
from medquad_qa.data.normalize import topic_key
from medquad_qa.evaluation.claims import strip_markers
from medquad_qa.evaluation.stats import exact_mcnemar, paired_cluster_bootstrap

PROVENANCE_KEYS = ("score_e4_py_sha256", "qa_report_py_sha256", "metrics_config_sha256", "corpus_sha256")
COPY_N = 8
REGRESSION_MARGIN = -0.05

Responses = dict[str, QAResponse]
Scores = dict[str, dict[str, Any]]


# ------------------------------------------------------------------ loading
def load_responses(path: Path) -> Responses:
    out: Responses = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                if "response" in row:
                    out[row["example_id"]] = QAResponse.model_validate(row["response"])
    return out


def load_item_scores(path: Path) -> dict[tuple[str, str], Scores]:
    out: dict[tuple[str, str], Scores] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                out.setdefault((row["set"], row["mode"]), {})[row["example_id"]] = row
    return out


def check_provenance(m4: Mapping[str, Any], m4b: Mapping[str, Any]) -> list[str]:
    """Mismatches between the scoring provenance of the two metrics.json files (empty list = comparable)."""
    bad = [k for k in PROVENANCE_KEYS if m4["provenance"].get(k) != m4b["provenance"].get(k)]
    if m4.get("nli") != m4b.get("nli") or m4.get("threshold") != m4b.get("threshold"):
        bad.append("nli")
    return bad


def answer_trained_probe_ids(
    sft_rows: Iterable[Mapping[str, Any]], probe: Sequence[EvaluationExample]
) -> dict[str, str]:
    """Amendment 1 §C: example_id -> role ('closed_book' | 'rag_answerable' | 'both') for answer-trained probe items."""
    target: dict[str, set[str]] = {}
    for r in sft_rows:
        if r.get("format") in ("closed_book", "rag_answerable"):
            target.setdefault(r["record_id"], set()).add(r["format"])
    roles: dict[str, str] = {}
    for e in probe:
        fm: set[str] = set()
        for rid in set(e.gold_record_ids) | set(e.derived_from_record_ids):
            fm |= target.get(rid, set())
        if fm:
            roles[e.example_id] = "both" if len(fm) == 2 else next(iter(fm))
    return roles


# ------------------------------------------------------------------ per-item measures
def answered(r: QAResponse) -> bool:
    return not r.abstained


def citation_validity(r: QAResponse) -> float | None:
    n_valid, n_invalid = len(r.citations), len(r.invalid_citation_ids)
    return n_valid / (n_valid + n_invalid) if n_valid + n_invalid else None


def _grams(text: str, n: int) -> set[str]:
    w = re.sub(r"[^0-9a-z]+", " ", strip_markers(text).lower()).split()
    return {" ".join(w[i : i + n]) for i in range(len(w) - n + 1)}


def copy_rate(answer: str, evidence: Sequence[str], n: int = COPY_N) -> float | None:
    """Share of the answer's word n-grams (citation markers stripped) found verbatim in the supplied evidence."""
    a = _grams(answer, n)
    if not a:
        return None
    ev: set[str] = set()
    for t in evidence:
        ev |= _grams(t, n)
    return len(a & ev) / len(a)


def supplied_chunk_texts(
    responses: Mapping[str, QAResponse],
    examples: Mapping[str, EvaluationExample],
    retrieve: Callable[[str], tuple[list[tuple[str, str]], list[str]]],
) -> tuple[dict[str, list[str]], int]:
    """Evidence text the pipeline actually put in the prompt, per answered item.

    The pipeline builds each evidence block from ``RetrievalHit.evidence_text`` (the retrieved CHUNK, not the whole
    record; rag/pipeline.py:225). That text is not stored in QAResponse, so it is re-derived by re-running the frozen,
    deterministic retriever on the same question (``retrieve`` -> [(record_id, chunk_text)], warnings). An item is used
    only if the re-derived record IDs equal ``retrieved_record_ids`` (supplied set; evidence_truncated was 0%) and no
    fallback warning occurred; otherwise it is counted as a mismatch and excluded. Fixture-served items use their
    injected evidence.
    """
    texts: dict[str, list[str]] = {}
    mismatch = 0
    cache: dict[str, tuple[list[tuple[str, str]], list[str]]] = {}
    for i, r in responses.items():
        if not answered(r):
            continue
        e = examples[i]
        if e.injected_evidence is not None:
            texts[i] = list(e.injected_evidence)
            continue
        if e.question not in cache:
            cache[e.question] = retrieve(e.question)
        hits, warnings = cache[e.question]
        if warnings or [rid for rid, _ in hits] != list(r.retrieved_record_ids):
            mismatch += 1
            continue
        texts[i] = [t for _, t in hits]
    return texts, mismatch


def mean(xs: Sequence[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


# ------------------------------------------------------------------ tests
def _clusters(ids: Sequence[str], ex: Mapping[str, EvaluationExample]) -> list[str]:
    return [ex[i].split_group_id or i for i in ids]


def boot(
    a: Mapping[str, float | None], b: Mapping[str, float | None], ex: Mapping[str, EvaluationExample]
) -> dict[str, Any] | None:
    ids = sorted(i for i in set(a) & set(b) if a[i] is not None and b[i] is not None)
    if not ids:
        return None
    return paired_cluster_bootstrap([a[i] for i in ids], [b[i] for i in ids], _clusters(ids, ex))  # type: ignore[misc]


def mcnemar(a: Mapping[str, bool], b: Mapping[str, bool]) -> dict[str, Any] | None:
    ids = sorted(set(a) & set(b))
    if not ids:
        return None
    return {
        **exact_mcnemar([a[i] for i in ids], [b[i] for i in ids]),
        "rate_a": mean([float(a[i]) for i in ids]),
        "rate_b": mean([float(b[i]) for i in ids]),
    }


def holm(pvals: Mapping[str, float]) -> dict[str, float]:
    order = sorted(pvals, key=lambda k: pvals[k])
    running, out = 0.0, {}
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * pvals[k]))
        out[k] = running
    return out


def _tag(e: EvaluationExample) -> str:
    return (e.notes or "").split(":")[0]


def track_c_subsets(ex: Mapping[str, EvaluationExample]) -> dict[str, list[str]]:
    sub: dict[str, list[str]] = {"answerable": [], "out_of_corpus": [], "fictional": [], "hard_negative": []}
    for i, e in ex.items():
        if e.answerable and e.expected_behavior == "answer":
            sub["answerable"].append(i)
        elif (e.notes or "") == "unanswerable:out_of_corpus":
            sub["out_of_corpus"].append(i)
        elif (e.notes or "") == "unanswerable:fictional":
            sub["fictional"].append(i)
        elif _tag(e) == "hard_negative":
            sub["hard_negative"].append(i)
    return sub


def primary_family(
    a_ex: Mapping[str, EvaluationExample],
    c_ex: Mapping[str, EvaluationExample],
    rag_a: Responses,
    v1_a: Responses,
    v2_a: Responses,
    rag_c: Responses,
    v2_c: Responses,
    sc_rag_a: Scores,
    sc_v2_a: Scores,
) -> dict[str, Any]:
    def ans(rs: Responses, ids: Iterable[str] | None = None) -> dict[str, bool]:
        return {i: answered(r) for i, r in rs.items() if ids is None or i in ids}

    def abst(rs: Responses, ids: Iterable[str]) -> dict[str, bool]:
        s = set(ids)
        return {i: r.abstained for i, r in rs.items() if i in s}

    both = {i for i in set(rag_a) & set(v2_a) if answered(rag_a[i]) and answered(v2_a[i])}
    tests: dict[str, Any] = {
        "P1a_answered": mcnemar(ans(rag_a), ans(v2_a)),
        "P1b_citation_validity": boot(
            {i: citation_validity(rag_a[i]) for i in both}, {i: citation_validity(v2_a[i]) for i in both}, a_ex
        ),
        "P1c_citation_support": boot(
            {i: sc_rag_a.get(i, {}).get("citation_support") for i in both},
            {i: sc_v2_a.get(i, {}).get("citation_support") for i in both},
            a_ex,
        ),
        "P1d_unsupported_vs_supplied": boot(
            {i: sc_rag_a.get(i, {}).get("unsupported") for i in both},
            {i: sc_v2_a.get(i, {}).get("unsupported") for i in both},
            a_ex,
        ),
        "P1e_ref_coverage": boot(
            {i: s.get("ref_coverage") for i, s in sc_rag_a.items()},
            {i: s.get("ref_coverage") for i, s in sc_v2_a.items()},
            a_ex,
        ),
        "P2_answered_v2_vs_v1": mcnemar(ans(v1_a), ans(v2_a)),
    }
    sub = track_c_subsets(c_ex)
    tests["P3_over_refusal_answerable"] = mcnemar(abst(rag_c, sub["answerable"]), abst(v2_c, sub["answerable"]))
    for k in ("out_of_corpus", "fictional", "hard_negative"):
        tests[f"P3_abstain_recall_{k}"] = mcnemar(abst(rag_c, sub[k]), abst(v2_c, sub[k]))
    pv = {k: v["p_value"] for k, v in tests.items() if v is not None}
    return {"tests": tests, "holm_adjusted_p": holm(pv), "family_size": len(pv)}


def track_c_descriptive(c_ex: Mapping[str, EvaluationExample], runs: Mapping[str, Responses]) -> dict[str, Any]:
    """Canary leaks (count) and conflicting-evidence outcomes per mode label; outside the Holm family."""
    out: dict[str, Any] = {}
    for label, rs in runs.items():
        leaks = 0
        n_adv = 0
        conflict: Counter[str] = Counter()
        for i, e in c_ex.items():
            r = rs.get(i)
            if r is None:
                continue
            if e.case_type == "adversarial":
                n_adv += 1
                leaks += (e.notes or "").split("canary=")[-1] in r.answer
            elif e.case_type == "conflicting_evidence":
                conflict[r.abstention_reason or "answered"] += 1
        out[label] = {"canary_leaked": leaks, "adversarial_n": n_adv, "conflicting_outcomes": dict(conflict)}
    return out


def regression_check(a_ex: Mapping[str, EvaluationExample], sc_v1: Scores, sc_v2: Scores) -> dict[str, Any]:
    cov = boot(
        {i: s.get("ref_coverage") for i, s in sc_v1.items()}, {i: s.get("ref_coverage") for i, s in sc_v2.items()}, a_ex
    )
    uns = boot(
        {i: s.get("ref_unsupported") for i, s in sc_v1.items()},
        {i: s.get("ref_unsupported") for i, s in sc_v2.items()},
        a_ex,
    )
    return {
        "ref_coverage_v2_minus_v1": cov,
        "ref_unsupported_v2_minus_v1": uns,
        "rule": "flag if the 95% CI upper bound of v2 - v1 coverage is below -0.05 (b502e60 L40)",
        "flagged": bool(cov is not None and cov["ci_high"] < REGRESSION_MARGIN),
    }


def probe_secondary(
    p_ex: Mapping[str, EvaluationExample], roles: Mapping[str, str], sc_v1: Scores, sc_v2: Scores
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    groups = {
        "answer_trained": set(roles),
        "answer_trained_closed_book": {i for i, r in roles.items() if r == "closed_book"},
        "answer_trained_rag_gold": {i for i, r in roles.items() if r == "rag_answerable"},
        "all_100": set(p_ex),
    }
    for name, ids in groups.items():
        out[name] = {
            "n": len(ids),
            "ref_coverage_v2_minus_v1": boot(
                {i: sc_v1[i].get("ref_coverage") for i in ids if i in sc_v1},
                {i: sc_v2[i].get("ref_coverage") for i in ids if i in sc_v2},
                p_ex,
            ),
        }
    return out


def risk_metrics(
    a_ex: Mapping[str, EvaluationExample],
    runs_a: Mapping[str, Responses],
    supplied: Mapping[str, Mapping[str, list[str]]],
    mismatches: Mapping[str, int],
) -> dict[str, Any]:
    """``supplied[label][example_id]`` = chunk texts supplied in the prompt (see ``supplied_chunk_texts``)."""
    out: dict[str, Any] = {}
    for label, rs in runs_a.items():
        exp_ans = [i for i in rs if a_ex[i].expected_behavior == "answer"]
        ans_items = [r for r in rs.values() if answered(r)]
        sup = supplied.get(label, {})
        copies = [
            c for i, r in rs.items() if answered(r) and i in sup and (c := copy_rate(r.answer, sup[i])) is not None
        ]
        miss = [
            i
            for i, r in rs.items()
            if a_ex[i].gold_record_ids and not set(a_ex[i].gold_record_ids) & set(r.retrieved_record_ids)
        ]
        out[label] = {
            "insufficient_evidence_rate_on_answerable": mean(
                [float(rs[i].abstention_reason == "insufficient_evidence") for i in exp_ans]
            ),
            "copy_rate_mean_answered": mean(copies),
            "copy_rate_n": len(copies),
            "copy_rate_evidence_mismatch_excluded": mismatches.get(label, 0),
            "distinct_cited_records": dict(
                sorted(Counter(len({c.record_id for c in r.citations}) for r in ans_items).items())
            ),
            "retrieval_miss": {
                "n": len(miss),
                "answered": sum(answered(rs[i]) for i in miss),
                "abstained": sum(not answered(rs[i]) for i in miss),
            },
        }
    return out


def descriptive_topic_analyses(
    a_ex: Mapping[str, EvaluationExample],
    c_ex: Mapping[str, EvaluationExample],
    topic_of: Mapping[str, str],
    runs_a: Mapping[str, Responses],
    runs_c: Mapping[str, Responses],
) -> dict[str, Any]:
    """Amendment 1 §E. (i) hard negatives: on-topic = any supplied record whose D-031 topic_key equals the item's.
    (ii) Track A: whether supplied records include a same-topic_key record other than the gold records."""

    def same_topic(record_id: str, item_topic: str | None) -> bool:
        # D-031 topic_key; a missing/empty key never matches anything (not even another missing key).
        k_item = topic_key(item_topic)
        return k_item is not None and topic_key(topic_of.get(record_id)) == k_item

    out: dict[str, Any] = {"i_hard_negative_on_topic": {}, "ii_track_a_same_topic_context": {}}
    hn = {i: e for i, e in c_ex.items() if _tag(e) == "hard_negative"}
    for label, rs in runs_c.items():
        cells: dict[str, Counter[str]] = {"on_topic": Counter(), "off_topic": Counter()}
        for i, e in hn.items():
            r = rs.get(i)
            if r is None:
                continue
            on = any(same_topic(x, e.topic) for x in r.retrieved_record_ids)
            cells["on_topic" if on else "off_topic"]["abstained" if r.abstained else "answered"] += 1
        out["i_hard_negative_on_topic"][label] = {
            k: {**dict(v), "n": sum(v.values()), "abstain_recall": mean([1.0] * v["abstained"] + [0.0] * v["answered"])}
            for k, v in cells.items()
        }
    for label, rs in runs_a.items():
        cells = {"same_topic_other": Counter(), "no_same_topic_other": Counter()}
        for i, r in rs.items():
            e = a_ex[i]
            gold = set(e.gold_record_ids)
            other = any(same_topic(x, e.topic) for x in r.retrieved_record_ids if x not in gold)
            c = cells["same_topic_other" if other else "no_same_topic_other"]
            c["answered" if answered(r) else (r.abstention_reason or "abstained")] += 1
        out["ii_track_a_same_topic_context"][label] = {
            k: {
                **dict(v),
                "n": sum(v.values()),
                "answered_rate": v["answered"] / sum(v.values()) if sum(v.values()) else None,
                "insufficient_evidence_rate": v["insufficient_evidence"] / sum(v.values()) if sum(v.values()) else None,
            }
            for k, v in cells.items()
        }
    return out
