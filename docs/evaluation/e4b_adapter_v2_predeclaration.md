# E4b: adapter v2 evaluation plan (pre-declared)

Owner: evaluation-safety-engineer. Written 2026-10-06, while E4 (v1) was still running. This was before v2 had been built or trained, and before any v2 output on any set existed. Basis: D-051 (adapter v2 approved in principle).

## Status of v1 and E4
- v1 (`…+lora:sft-main-20261006-081347-3f16e30e`) stays the promoted adapter.
- The E4 report and all E4 artifacts stay tied to v1's `model_version`.
- v2 is reported separately as **E4b**, a follow-up. It does not replace E4 numbers.

## Gates before any v2 TEST run
1. `check_protected_leakage.py` passes on v2's `sft_record_ids.jsonl`. Every `record_id` and every `evidence_record_ids` entry counts as trained-on: it must be in a train group, it must not share a group with any dev or test eval item, and it must carry the frozen `split_version`. The same check runs on any v2 validation-build ID file (`--threshold-ids`).
2. v2 is tuned and selected on validation and DEV only (D-018/D-035), e.g. checkpoint, mix ratio, or any decoding change. TEST is scored once.
3. Pipeline code, prompts, safety rules, retriever, index and gate are unchanged from freeze `1c84b60`. Only the adapter differs. `e4_env.sh` is reused with `finetuned_model_version` set to v2. Every other `--expect` value stays the same.
4. Step 0 on DEV, 15 items, `finetuned` and `finetuned_rag`, with the same checks as E4 Step 0. It must show that v2 emits valid `[E#]` citations on DEV.

## Runs (v2 only; base and rag outputs are reused from E4, same frozen state)
| Set | Modes (v2) | Items |
|---|---|---|
| test_track_a | finetuned, finetuned_rag | 300 × 2 |
| test_track_c | finetuned_rag; finetuned on personal-advice + controls only (as in E4 tier 4) | 240 + 80 |
| train_probe | finetuned | 100 |

Estimate: about 1,020 pipeline calls at about 8 s each, so about 2.3 h. It runs alone (D-045) and uses the E4 harness unchanged.

## Pre-declared comparisons
**Primary.** Holm correction within this family. Paired cluster bootstrap (10k resamples, seed 20261006, clusters = `split_group_id`). Exact McNemar for binary outcomes.
- P1, on Track A: finetuned_rag v2 vs rag (base).
  - (a) answered rate
  - (b) citation validity
  - (c) citation support (NLI)
  - (d) unsupported-claim rate against the supplied evidence
  - (e) reference coverage, where abstained = 0
- P2, on Track A: finetuned_rag v2 vs v1, on answered rate. v1 is expected to be about 0 because of missing citations, so this documents the fix.
- P3, on Track C: finetuned_rag v2 vs rag.
  - over-refusal on answerable items and general controls
  - abstention recall on unanswerable items, fictional items and hard negatives (each reported separately)
  - injection canary leaks (count; any leak is reported as a finding)
  - conflicting-evidence behaviour

**Regression check, not in the primary family.** finetuned (closed-book) v2 vs v1 on Track A, using reference coverage and the reference-unsupported claim rate. v2 has about half as many closed-book examples as v1. A regression is flagged if the 95% CI upper bound of v2 − v1 coverage is below −0.05.

**Secondary.** train_probe memorization, finetuned v2 vs v1 (reference coverage). Also the resource-list-gold subgroup, as in E4.

## Pre-declared risk metrics
- **Over-refusal.** Sentinel targets are about 16% of the RAG training examples. Report the `insufficient_evidence` rate on answerable Track A items, v2 vs rag.
- **Extractive copying.** Report the copy rate: the share of the answer's word 8-grams (citation markers stripped) that appear verbatim in the supplied evidence. Show it for v2 finetuned_rag vs rag, on answered items. This describes the answers; it is not a quality score.
- **Gold always present in training.** Track C hard negatives and unanswerable items measure whether v2 answers anyway when no gold is supplied. Retrieval misses on Track A are reported with the outcome in each case: v2 answered or abstained.
- **Single-source targets.** Report the distribution of the number of distinct cited records per answer, v2 vs rag.

## Reporting
- E4b results go next to E4 and are labelled with v2's `model_version`.
- Nothing is labelled human-reviewed. AI-agent rubric checks, if any are run, are labelled `ai_agent`.
- Any deviation from this plan is listed explicitly as a deviation, with the reason.
