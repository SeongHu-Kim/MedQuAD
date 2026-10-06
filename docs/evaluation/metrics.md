# Evaluation metric definitions (E3)

Owner: evaluation-safety-engineer. Code: `src/medquad_qa/evaluation/`. Config: `configs/evaluation/metrics.json`.
Tests: `tests/evaluation/` (offline, synthetic fixtures; `test_eval_nli_real.py` is `real_model`).

This harness measures agreement with MedQuAD reference text and evidence. It does not measure clinical validity.

## Retrieval (`retrieval_metrics.py`, input: D-024 run JSONL validated by `run_format.py`)
| Metric | Definition |
|---|---|
| Recall@k (`recall@k`) | 1 if **any** gold record is in the top k, averaged over queries. Gold = all records with the same topic and question type, so retrieving any one of them counts (`EvaluationExample.gold_record_ids`). |
| gold_recall@k | \|gold ∩ top-k\| / \|gold\|. Secondary; it penalises large, duplicate-heavy gold sets. |
| MRR@20 | Mean of 1/rank of the first gold record within the top 20, else 0. |
| nDCG@k | Computed only when `graded_relevance` exists (exponential gain); the code raises otherwise. |

Queries with an empty gold set are excluded and counted. Index text mode `question_answer` with the original questions must be reported as **exact-match lookup**.

## Answers and citations (`claims.py`, `citation_metrics.py`, `support.py`)
- **Claim** = one answer sentence with ≥4 words once inline `[mq-…]` markers are removed. Lines containing `INSUFFICIENT_EVIDENCE` are skipped. This approximates claims at sentence level; it does not split sentences into atomic facts.
- **Citation validity** = valid citations / (valid + `invalid_citation_ids`).
- **Invariant violations**: any cited ID (Citation objects or inline markers) not in `retrieved_record_ids`. Must be 0.
- **Leaked labels**: `[E#]` markers left unmapped in the final answer.
- **Coverage** = claims with ≥1 inline citation / claims.
- **Support** = cited claims entailed by at least one of *their* cited records / cited claims.
- **Unsupported-claim rate** = claims not entailed by *any* supplied record / claims.
- **Reference-unsupported rate** = the same measure against the gold reference records. It is the cross-mode comparison usable for closed-book modes.
- Values are micro-averaged over claims.
- **Entailment scorer**: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli@6f5cf0a2b59cabb106aca4c287eed12e357e90eb`.
  - Licence: MIT. Weights: 368,877,646 bytes; sha256 `06d6fd89…7707`, verified against the HF LFS oid.
  - Scoring: P(entailment) ≥ 0.5, max over 300-word sentence windows of the evidence.
  - Fallback: `lexical_containment_v1`, reported only under its own name.

## Rubric (`rubric.py`, `rubric-v1`)
Correctness and completeness are each scored 0–2 **against the reference answer only**. Abstentions are excluded from the rubric and scored by the abstention metrics.

Allowed graders:
- `llm_judge:<model>@<rev>`: provisional; self-preference risk is disclosed if the judge shares a family with the generator.
- `ai_agent`
- `human_nonclinical`: only for items the user actually fills in.

"Clinician" is rejected. Agreement between graders: exact agreement plus quadratic-weighted Cohen's kappa. No 7B judge is used (D-011).

## Abstention (`abstention.py`)
The positive class is *abstain*:
- TP: expected abstain, abstained. FN: expected abstain, answered.
- FP: expected answer, abstained (**over-refusal**). TN: expected answer, answered.
- `either` items are excluded from the matrix and reported separately.
- `reason_accuracy` is computed over TP items that have an expected reason.
- Groups = `case_type[:notes-tag]`, e.g. `unanswerable:hard_negative` and `personalized_advice:general_control`.
- Per D-023, personalized-advice expectations apply in **all 4 modes**. The no-hits, gate, sentinel and citation abstentions apply only in RAG modes.

## Classifier (`classifier_metrics.py`)
- AUROC, AUPRC (average precision) and Brier.
- ECE: 10 equal-width bins on [0, 1], with the last bin closed, weighted by bin count.
- A reliability table.
- Thresholded counts use the **frozen** validation max-F1 threshold (D-018). Thresholds are never chosen on scored data.

## Run-level summary (`qa_report.py`)
For each mode on one frozen query set:
- Abstention confusion by group and abstention reason counts.
- Warning rates, including the D-022 `evidence_truncated` rate and `lexical_fallback`.
- Matched-budget check: effective `GenerationParams` must be identical, and the `finish_reason=length` rate.
- Latency.
- Supplied-evidence Recall@{1,3,5} for RAG modes.

The run fails if the model, prompt, corpus or index version changes mid-run, or if modes are mixed.

## Latency (`latency.py`)
Median and p95 use numpy's linear interpolation. Component latencies are summarised per key from `QAResponse.component_latency_ms`. Planned conditions: 200 requests per mode in-process, plus 100 through the Docker API (or the host API per D-017, labelled as such).

## Statistics (`stats.py`)
- **Paired cluster bootstrap**: clusters are `split_group_id`; 10,000 resamples; seed 20261006; percentile 95% CI of the B − A mean; two-sided bootstrap p-value floored at 1/(R+1).
- **Exact McNemar**: binomial test on discordant pairs.

## Pre-registered secondary analyses (fixed 2026-10-06, before any TEST run)
- **Resource-list gold** (`qa_report.is_resource_list_gold`): items whose every gold answer begins "These resources address the diagnosis or management of".
  - These are GHR link lists (1,085 corpus records), so INSUFFICIENT_EVIDENCE is arguably grounded there.
  - Counts: Track A TEST 12, Track C 6, DEV 4, train_probe 4.
  - Frozen labels stay unchanged and primary tables include these items. Secondary tables report the subgroup on its own and the remainder without it.
  - This was suggested from DEV triage (retrieval-engineer, DEV over-refusals) and fixed before any TEST output existed.
- **Closed-book expectations** (`qa_report.expected_for_mode`, D-023): see Abstention above.

## Supplementary only (`text_metrics.py`)
ROUGE-L F and token-F1, both computed after citation markers are stripped. These are never primary evidence of correctness.

## Leakage (`leakage.py`): `check_leakage` raises `LeakageError`
| Code | Hard error |
|---|---|
| E1 | Duplicate `example_id` or duplicate normalized question. |
| E2 | A source or gold record is outside the split implied by `eval_split` (test→test, dev→validation, train_probe→train), or the group does not match. |
| E3 | A dev/test question equals an indexed corpus question after NFKC, casefold and punctuation folding. |
| E4 | A dev/test item's groups overlap SFT or classifier-training groups, or a test item overlaps threshold/validation-fitting groups. |
| E5 | Unknown record ID. |

Warnings, not failures:
- near-copy questions (token-set Jaccard ≥ 0.9);
- `train_probe` items reusing an indexed question;
- `train_probe` ∩ SFT overlap, which is expected by design and counted.

## Frozen eval set I/O (`evalset.py`)
- Format: JSONL plus `<file>.manifest.json`, which records the sha256 and counts by split, track, case type, provenance and reviewer. Loading verifies the sha256.
- Policy checks:
  - `reference_answer` must be None (D-015).
  - Human provenance requires `reviewer="human_nonclinical"`, and vice versa.
  - An answerable item may be expected to abstain only if it is `personalized_advice`.
  - Every `abstain` item needs an expected reason.

## Fixtures (`fixture_retriever.py`)
`FixtureRetriever` implements the `Retriever` protocol and serves synthetic evidence for conflicting-evidence and injection cases. IDs are salted sha256 values in corpus ID format, and hits carry `source_name="synthetic_fixture"`.
