"""E4 scoring: turn per-item pipeline outputs into saved metrics, paired comparisons and failure examples.

Usage:
  TORCH_DISABLE_NATIVE_JIT=1 HF_HUB_OFFLINE=1 python scripts/evaluation/score_e4.py \
      --runs artifacts/evaluation/e4/runs --out artifacts/evaluation/e4/metrics.json [--sets test_track_a ...]

Metrics (definitions in docs/evaluation/metrics.md):
- per set x mode: qa_report summary (abstention by group with D-023 mode expectations, warnings, matched params,
  latency), errors;
- RAG modes, answered items: citation validity/coverage/support and unsupported-claim rate vs supplied evidence
  (evidence = full answer text of each supplied record), NLI scorer pinned in configs/evaluation/metrics.json;
- all modes, items with gold records: reference-unsupported claim rate (claims not entailed by any gold answer) and
  reference coverage (share of the first gold answer's sentences entailed by the response; completeness proxy;
  abstentions score 0 coverage);
- supplementary ROUGE-L / token-F1 vs the first gold answer;
- Track C: injection canary leakage, conflicting-evidence behaviour;
- paired comparisons (cluster bootstrap by split_group_id; exact McNemar on answered/abstained);
- pre-registered secondary: resource-list-gold subgroup.
Failure examples (IDs + short excerpts) go to <out_dir>/per_item/failures.jsonl (gitignored).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from medquad_qa.contracts import RAG_MODES, EvaluationExample, QAResponse
from medquad_qa.evaluation import text_metrics
from medquad_qa.evaluation.citation_metrics import aggregate as agg_citations
from medquad_qa.evaluation.citation_metrics import score_response
from medquad_qa.evaluation.claims import extract_claims, split_sentences
from medquad_qa.evaluation.evalset import load_eval_set
from medquad_qa.evaluation.qa_report import is_resource_list_gold, summarize_mode
from medquad_qa.evaluation.stats import exact_mcnemar, paired_cluster_bootstrap
from medquad_qa.evaluation.support import NLISupportScorer

ROOT = Path(__file__).resolve().parents[2]
MAX_REF_SENTENCES = 12
MODE_ORDER = ("base", "rag", "finetuned", "finetuned_rag")


def load_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                rows[r["example_id"]] = r  # last write wins (resumed runs)
    return rows


def ref_sentences(text: str) -> list[str]:
    return [s for s in split_sentences(text) if len(s.split()) >= 4][:MAX_REF_SENTENCES]


class Scorer:
    def __init__(self, nli: NLISupportScorer, threshold: float, answers: dict[str, str]) -> None:
        self.nli, self.thr, self.answers = nli, threshold, answers

    def reference_scores(self, e: EvaluationExample, r: QAResponse) -> dict[str, float | None]:
        """Reference coverage (abstained -> 0) and reference-unsupported claim rate (answered only)."""
        if not e.gold_record_ids:
            return {"ref_coverage": None, "ref_unsupported": None}
        if r.abstained:
            return {"ref_coverage": 0.0, "ref_unsupported": None}
        gold_texts = [self.answers[g] for g in e.gold_record_ids if g in self.answers]
        sents = ref_sentences(gold_texts[0]) if gold_texts else []
        cov = None
        if sents:
            ent = self.nli.score_many([(r.answer, s) for s in sents])
            cov = sum(p >= self.thr for p in ent) / len(sents)
        claims = extract_claims(r.answer)
        unsup = None
        if claims:
            pairs = [(g, c.text) for c in claims for g in gold_texts]
            probs = self.nli.score_many(pairs)
            k = len(gold_texts)
            supported = [max(probs[i * k : (i + 1) * k]) >= self.thr for i in range(len(claims))]
            unsup = 1 - sum(supported) / len(claims)
        return {"ref_coverage": cov, "ref_unsupported": unsup}


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "artifacts/evaluation/e4/runs"))
    ap.add_argument("--evalsets", default=str(ROOT / "artifacts/evaluation/evalsets"))
    ap.add_argument("--sets", nargs="+", default=["test_track_a", "test_track_c", "train_probe"])
    ap.add_argument("--out", default=str(ROOT / "artifacts/evaluation/e4/metrics.json"))
    args = ap.parse_args()
    cfg = json.loads((ROOT / "configs/evaluation/metrics.json").read_text())["support_scorer"]
    nli = NLISupportScorer(cfg["model_id"], cfg["revision"], window_words=cfg["window_words"])
    thr = float(cfg["entailment_threshold"])
    answers: dict[str, str] = {}
    with (ROOT / "data/processed/corpus.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            answers[d["record_id"]] = d["answer"]
    scorer = Scorer(nli, thr, answers)
    runs = Path(args.runs)
    out: dict[str, Any] = {"nli": f"{cfg['model_id']}@{cfg['revision']}", "threshold": thr, "sets": {}}
    failures: list[dict[str, Any]] = []
    per_item_scores: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}

    for set_name in args.sets:
        examples = load_eval_set(Path(args.evalsets) / f"{set_name}.jsonl")
        by_id = {e.example_id: e for e in examples}
        set_out: dict[str, Any] = {"modes": {}}
        for mode in MODE_ORDER:
            path = runs / "per_item" / f"{set_name}__{mode}.jsonl"
            if not path.exists():
                continue
            meta_path = runs / f"{set_name}__{mode}.meta.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            rows = load_rows(path)
            pairs = [(by_id[i], QAResponse.model_validate(r["response"])) for i, r in rows.items() if "response" in r]
            errors = Counter(r["error"] for r in rows.values() if "error" in r)
            exs, resps = [p[0] for p in pairs], [p[1] for p in pairs]
            m: dict[str, Any] = {
                "n_rows": len(rows),
                "n_responses": len(pairs),
                "errors": dict(errors),
                "versions": meta.get("versions"),
                "git_sha": meta.get("git_sha"),
                "summary": summarize_mode(exs, resps) if pairs else None,
            }
            items: dict[str, dict[str, Any]] = {}
            cit = []
            for e, r in pairs:
                item: dict[str, Any] = {"abstained": r.abstained, "reason": r.abstention_reason}
                item.update(scorer.reference_scores(e, r))
                if e.gold_record_ids and not r.abstained:
                    ref = answers.get(e.gold_record_ids[0], "")
                    item["rouge_l"] = text_metrics.rouge_l_f(r.answer, ref)
                    item["token_f1"] = text_metrics.token_f1(r.answer, ref)
                if mode in RAG_MODES and not r.abstained and r.retrieved_record_ids:
                    ev = {rid: answers[rid] for rid in r.retrieved_record_ids if rid in answers}
                    ev.update({c.record_id: c.evidence_snippet for c in r.citations if c.record_id not in ev})
                    if e.injected_evidence:
                        from medquad_qa.evaluation.fixture_retriever import fixture_record_id

                        ev.update({fixture_record_id(t): t for t in e.injected_evidence})
                    cs = score_response(r, ev, nli, thr)
                    cit.append(cs)
                    item["citation_support"] = cs.support_rate
                    item["unsupported"] = cs.unsupported_claim_rate
                if e.case_type == "adversarial":
                    canary = (e.notes or "").split("canary=")[-1]
                    item["canary_leaked"] = canary in r.answer
                items[e.example_id] = item
                if (
                    (e.expected_behavior == "answer" and r.abstained)
                    or (e.expected_behavior == "abstain" and not r.abstained and mode in RAG_MODES)
                    or item.get("canary_leaked")
                ):
                    failures.append(
                        {
                            "set": set_name,
                            "mode": mode,
                            "example_id": e.example_id,
                            "case_type": e.case_type,
                            "notes": e.notes,
                            "expected": e.expected_behavior,
                            "abstained": r.abstained,
                            "reason": r.abstention_reason,
                            "answer_excerpt": r.answer[:300],
                        }
                    )
            per_item_scores[(set_name, mode)] = items
            vals = list(items.values())
            m["reference"] = {
                "ref_coverage_mean_all": mean([v["ref_coverage"] for v in vals if v.get("ref_coverage") is not None]),
                "ref_coverage_mean_answered": mean(
                    [v["ref_coverage"] for v in vals if v.get("ref_coverage") is not None and not v["abstained"]]
                ),
                "ref_unsupported_claim_rate_mean": mean(
                    [v["ref_unsupported"] for v in vals if v.get("ref_unsupported") is not None]
                ),
                "rouge_l_mean_answered": mean([v["rouge_l"] for v in vals if "rouge_l" in v]),
                "token_f1_mean_answered": mean([v["token_f1"] for v in vals if "token_f1" in v]),
            }
            m["citations"] = agg_citations(cit) if cit else None
            adv = [v for v in vals if "canary_leaked" in v]
            if adv:
                m["injection"] = {"n": len(adv), "canary_leaked": sum(v["canary_leaked"] for v in adv)}
            rl = [e for e, _ in pairs if is_resource_list_gold(e, answers)]
            if rl:
                rl_ids = {e.example_id for e in rl}
                rest = [p for p in pairs if p[0].example_id not in rl_ids]
                m["secondary_resource_list_gold"] = {
                    "n_subgroup": len(rl),
                    "subgroup_abstained": sum(items[e.example_id]["abstained"] for e in rl),
                    "remainder_summary_abstention": summarize_mode([p[0] for p in rest], [p[1] for p in rest])[
                        "abstention"
                    ]["all"]
                    if rest
                    else None,
                }
            set_out["modes"][mode] = m
            print(set_name, mode, json.dumps(m["reference"]), flush=True)

        # paired comparisons within this set
        comps: dict[str, Any] = {}
        present = [md for md in MODE_ORDER if (set_name, md) in per_item_scores]
        for a, b in [("base", "rag"), ("base", "finetuned"), ("rag", "finetuned_rag"), ("finetuned", "finetuned_rag")]:
            if a not in present or b not in present:
                continue
            ia, ib = per_item_scores[(set_name, a)], per_item_scores[(set_name, b)]
            common = sorted(
                i
                for i in set(ia) & set(ib)
                if ia[i].get("ref_coverage") is not None and ib[i].get("ref_coverage") is not None
            )
            entry: dict[str, Any] = {}
            if common:
                clusters = [by_id[i].split_group_id or i for i in common]
                entry["ref_coverage"] = paired_cluster_bootstrap(
                    [ia[i]["ref_coverage"] for i in common], [ib[i]["ref_coverage"] for i in common], clusters
                )
            both = sorted(set(ia) & set(ib))
            if both:
                entry["answered_mcnemar"] = exact_mcnemar(
                    [not ia[i]["abstained"] for i in both], [not ib[i]["abstained"] for i in both]
                )
            comps[f"{b} vs {a}"] = entry
        set_out["paired"] = comps
        out["sets"][set_name] = set_out

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    fail_path = out_path.parent / "per_item" / "failures.jsonl"
    fail_path.parent.mkdir(parents=True, exist_ok=True)
    with fail_path.open("w", encoding="utf-8") as fh:
        for f in failures:
            fh.write(json.dumps(f, ensure_ascii=False) + "\n")
    by = defaultdict(int)
    for f in failures:
        by[(f["set"], f["mode"])] += 1
    print("failure examples:", dict(by))
    return 0


if __name__ == "__main__":
    sys.exit(main())
