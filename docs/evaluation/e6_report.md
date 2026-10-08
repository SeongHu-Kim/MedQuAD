# E6: independent evaluation report

Owner: evaluation-safety-engineer (an AI agent). Written 2026-10-08, after D-065 (`69acefd`). Status: draft for the lead and the user.

This report follows the outline the user approved. Every number is copied from a committed artifact or decision row, and the source path and key are given next to it. Where a value is computed here from cited counts, the text says so. It contains no per-item text, no probe content and no identifiers from private files. Nothing in it is human-reviewed. Where an AI agent applied a label, the label is `ai_agent`.

Notation:
- `[a, b]` is a 95% interval. Wilson intervals are used for single proportions. Paired differences use the paired cluster bootstrap (10,000 resamples, seed 20261006, clusters = `split_group_id`).
- "Holm" means Holm-adjusted p within the pre-declared family named in the text.
- "Track A" is `test_track_a`, "Track C" is `test_track_c`, and "probe" is `train_probe`.

---

## 1. Scope and claims boundary

- This is a **non-clinical research prototype** for medical information question answering over MedQuAD (AgentTeamsPrompt §1). It has **not been clinically validated**. It has **not been published**, has **not been deployed publicly**, and has **not been used with real users**. The demo and deployment stay blocked (D-052, restated in D-065).
- Retrieval scores, answerability-gate scores, NLI scores and the rates in this report are **not medical confidence** and do not predict patient outcomes.
- "Unsupported by the reference answer" means that the NLI model did not find the claim entailed by MedQuAD's reference answer. It does **not** mean "medically false" (D-052).
- All evaluation questions and labels are synthetic: either template-generated (`synthetic_rule`) or AI-written without human review (`llm_generated_unreviewed`). No evaluation label in this project is `human_reviewed`. The optional user spot-check was never used.
  - The only human review in the project is the user's review of 40 **training** prompts from the rejected v2b build (D-053). It is not an evaluation label set.
- Resource budget: local GB10 only, with no paid compute (D-005).

## 2. Data and leakage control

**Corpus.** `medquad-1.0.0-86e384302357` (`data/manifests/corpus_manifest.json`, sha256 `f0f6b09e…`).
- `source_rows` 16,412 → `records` 16,336, with 76 rows excluded (`excluded_by_reason`: 48 exact duplicate rows, 23 non-informative answers, 5 empty answers).
- `medquad.csv` is unchanged: its sha256 `f9fe9e60…` is recorded as `source_sha256`.

**Split.** `split-20261006-2f0fb25ee6d8` was frozen in D-034.
- Records: 13,014 train / 1,665 validation / 1,657 test.
- Groups: 3,338 / 428 / 446 (`split_manifest.meta.json` `groups_per_split`; `report_retest_2f0fb25ee6d8.json` `records_per_split`).
- Grouping method: groups are derived from **content and topic, not documents**, because the export has no source URLs or document IDs. The manifest's own wording is: "Leakage is reduced, not proven absent" (`grouping_disclosure`).

**F-001 and its fix.**
- My pre-freeze verification failed (`artifacts/evaluation/split_verification/report_prefreeze.json`, `passed: false`). The cross-split overlaps were:
  - answer near-duplicates: 33
  - normalised questions: 10
  - folded topics: 8
  - family-content exact answers: 4
- The data rules were fixed (D-031: Unicode/accent folding of keys, family-aware boilerplate, rare-shingle cutoff df ≤ 25, bracketed aliases linking groups).
- My retest on the frozen split passed (`report_retest_2f0fb25ee6d8.json`, `passed: true`): every `cross_split_overlap_counts` value is 0, and every data steward `blocking_checks` value is 0 (`data/manifests/leakage_report.json`).

**Residual risks** (diagnostics, not blocking):
- 67 topic families are not grouped across splits (`report_retest_…json` `diagnostic_counts.topic_family`).
- 671 residual answer pairs cross splits at Jaccard 0.5–0.8 (`leakage_report.json` `diagnostics.residual_pairs_cross_split`; 2,215 residual pairs in total).
- 7 generic boilerplate answers appear in more than one split (`boilerplate_answers_cross_split`).

**Protected-set leakage at the E4 gate.** `artifacts/evaluation/leakage/protected_check_m4.json` passed (`passed: true`, `errors: []`) for the v1 SFT record IDs (12,636), the classifier training pairs (12,636 records) and the threshold set (1,610).
- `train_probe_sft_overlap` 100 is by design: the probe consists of trained-on items.
- The v2c build's gate also passed: 9,697 SFT rows, `train_probe_sft_overlap` 96 (`docs/evidence/D-053/gate_sft-mix-v2c-20261007-094359.stdout.txt`, listed in that folder's `SHA256SUMS`).

## 3. Evaluation design

**Frozen sets** (`artifacts/evaluation/evalsets/build_report.json`, sha256 `05dbfad8…`; each set has a `.manifest.json`):

| Set | n | Case types | Question provenance |
|---|---|---|---|
| dev | 60 | 40 answerable, 10 personal advice, 10 unanswerable | 35 synthetic_rule, 25 llm_generated_unreviewed |
| test_track_a | 300 | 300 answerable | 200 synthetic_rule, 100 llm_generated_unreviewed |
| test_track_c | 240 | 40 answerable general controls, 40 personal advice, 50 unanswerable (20 out-of-corpus, 30 fictional; D-053 §9), 50 hard negatives, 20 adversarial (injection), 20 ambiguous, 20 conflicting | 90 synthetic_rule, 150 llm_generated_unreviewed |
| train_probe | 100 | 100 answerable (trained-on) | 100 synthetic_rule |

**Modes.** The four modes are base, rag, finetuned (closed-book LoRA) and finetuned_rag. Personalized-advice safety rules run in all four modes. Retrieval, the answerability gate, the INSUFFICIENT_EVIDENCE sentinel and the citation checks are RAG-only (D-023). On Track C, the closed-book modes run only on the 40 personal-advice items and the 40 general controls (n = 80).

**Held-out policy.** D-035 says owners must not read `test_*` or `train_probe`, and the only tuning set is DEV (D-018). Breaches are disclosed:
- **D-053 §9:** the lead read `train_probe.jsonl` (schema and record-ID fields) and computed counts from `test_track_c.jsonl`.
- **D-061 and D-062:** one Track C ambiguous item was disclosed to an owner through a bank-row collision (§8.3). It is **excluded from all ambiguous-case reporting**, so this report gives no per-outcome ambiguous counts.

**Pre-declarations.**
- E4: `docs/evaluation/metrics.md`, the Holm families (8 / 8 / 2), and Step 0 on DEV (D-051).
- E4b: `docs/evaluation/e4b_adapter_v2_predeclaration.md` (`b502e60`, sha256 `f50fd1b2…`) and `e4b_predeclaration_amendment_1.md` (`19a90e64…`). The pre-declaration's author had seen the DEV Step 0 result but no TEST output (D-055).
- Remediation: `docs/evaluation/remediation_f009_f010_f011_acceptance.md` (`f4bf7e9`, sha256 `c3ab9adb…`) and the fresh probes (D-059), both committed before any remediation code existed.

## 4. Retrieval (R4)

Source: `artifacts/evaluation/retrieval/scores_test_track_a.json` (sha256 `46067cb1…`), key `runs.<variant>.metrics`. Track A has n = 300, every variant has 0 error rows and 0 lexical fallbacks, and the device is CPU.

| Variant | R@1 | R@5 | R@20 | MRR@20 |
|---|---|---|---|---|
| bm25:answer | 0.503 | 0.743 | 0.863 | 0.604 |
| bm25:qa | 0.607 | 0.843 | 0.937 | 0.707 |
| dense:answer | 0.673 | 0.890 | 0.927 | 0.765 |
| **dense:qa** (frozen, D-037) | **0.830** | **0.950** | **0.983** | **0.884** |
| hybrid_rrf:answer | 0.643 | 0.853 | 0.913 | 0.734 |
| hybrid_rrf:qa | 0.747 | 0.920 | 0.977 | 0.823 |
| hybrid_rrf+ce:answer | 0.693 | 0.850 | 0.913 | 0.766 |
| hybrid_rrf+ce:qa | 0.693 | 0.870 | 0.977 | 0.771 |

- **Paired R@5 differences** (`paired_comparisons`):
  - dense:qa − dense:answer +0.060 [+0.033, +0.087]
  - dense:qa − bm25:qa +0.107 [+0.073, +0.143]
  - hybrid_rrf:qa − dense:qa −0.030 [−0.050, −0.013]
  - hybrid_rrf+ce:qa − dense:qa −0.080 [−0.113, −0.050]. Neither the hybrid nor the cross-encoder variants improved on dense:qa.
- **Q+A index disclosure (D-037).** The `qa` indexes contain the original MedQuAD questions, and Track A questions are paraphrases of them. Headline numbers are therefore labelled "Q+A index", and the answer-only variants are reported beside them.
- **Question provenance** (`scores_by_question_provenance.json`): dense:qa R@5 is 0.965 on the 200 template questions and 0.92 on the 100 AI-written questions. For bm25:answer the figures are 0.785 and 0.66. Template paraphrases are easier, so the 0.95 headline is optimistic for unseen phrasing.
- **GPU vs CPU** (`gpu_vs_cpu_agreement.json`): all 8 runs have identical top-1, top-5 and top-20 rankings on all 300 items.
- `latency_valid` is `false` in this file, so no retrieval latency is reported from it.

## 5. Answerability classifier

Source: `artifacts/evaluation/classifier/metrics.json` (sha256 `adfdf880…`), key `models.<model>.splits.test`. The test set has 3,876 pairs (1,938 positive).

| Model | AUROC (all) | F1 (all) | AUROC, excl. own answer (n 2,264) | AUROC easy random | AUROC lexical-hard BM25 | AUROC same topic, different question type |
|---|---|---|---|---|---|---|
| lexical_lr (served gate, threshold 0.307) | 0.826 | 0.779 | 0.784 | 0.944 | 0.824 | **0.536** |
| keras_bigru (not served) | 0.961 | 0.891 | 0.934 | 0.967 | 0.914 | 0.914 |

- **The served LR gate is close to chance on same-topic negatives:** AUROC 0.536, with 479 of 564 such negatives above threshold (`by_negative_type.same_topic_diff_qtype.fp`/`tn`).
- **In the served pipeline,** E4 Track C rag hard negatives (n = 50) mostly passed the gate (`hard_negative_gate_vs_sentinel`: `gate_passed` 49, `gate_rejected` 1). The generator's INSUFFICIENT_EVIDENCE sentinel then caught 41 of them, and 8 were answered (`artifacts/evaluation/e4/metrics.json`, `sets.test_track_c.modes.rag.hard_negative_gate_vs_sentinel`).
- **Abstention on hard negatives therefore relies mainly on the generator, not the gate.**

## 6. E4: four-mode comparison (adapter v1)

E4 ran on freeze `1c84b60` and produced 2,040 per-item rows with 0 errors (D-055). Source: `artifacts/evaluation/e4/metrics.json` (sha256 `191b790b…`), scored by `score_e4.py` `f0ea7542…` and `qa_report.py` `d1f7252f…`, as recorded in `provenance`.

Frozen versions:
- retriever `dense:qa`, index `dense-qa-dc6b6a345fca`
- prompt `rag-v1+df593554`
- safety `safety-v2+43836b6c`
- gate `answerability-lexlr@3defcd00a31f:maxf1-val:0.307172`
- base `Qwen/Qwen3-4B-Instruct-2507@cdbee75f`
- adapter v1 `sft-main-20261006-081347-3f16e30e`
- greedy decoding, `max_new_tokens` 256

### 6.1 Track A (n = 300, answerable)

From `sets.test_track_a.modes.<mode>`:

| Mode | Answered | Reference coverage (all; abstained = 0) | Reference-unsupported claim rate (answered) | Citation validity / support | Finish by length | Item latency median / p95 (harness) |
|---|---|---|---|---|---|---|
| base | 300 | 0.384 | 0.714 | n/a | 0.49 | 12,810 / 13,492 ms |
| rag | 243 | 0.379 | 0.268 | 1.000 / 0.857 | 0.018 | 5,485 / 12,317 ms |
| finetuned | 300 | 0.354 | 0.433 | n/a | 0.32 | 6,933 / 14,965 ms |
| finetuned_rag | **7** | 0.023 | 0.071 (n = 7) | 1.000 / 1.000 (n = 7) | 0.26 | 10,185 / 16,276 ms |

- rag abstained on 57 of 300 answerable items: over-refusal 0.19 (29 insufficient_evidence, 27 no_relevant_evidence, 1 missing_citations).
- finetuned_rag answered 7 of 300; the other 266 were `missing_citations`, an over-refusal of 0.977 (D-052). This is because the closed-book v1 adapter emits no `[E#]` labels, which DEV Step 0 found before any TEST output (D-051). Every finetuned_rag answer-conditional figure above rests on 7 items.
- **Primary family** (8 tests, Holm; `paired`, `paired_holm_adjusted_p_primary`; "b − a" in the order written):

| Comparison | Answered (McNemar; Holm p) | Reference coverage diff [95% CI] (Holm p) |
|---|---|---|
| rag vs base | rag answered 57 fewer (Holm 8.3e-17) | −0.005 [−0.051, +0.042] (Holm 1.0) |
| finetuned vs base | no difference (both 300; Holm 1.0) | −0.030 [−0.075, +0.015] (Holm 0.605) |
| finetuned_rag vs rag | 236 fewer (Holm 1.3e-70) | −0.355 [−0.398, −0.313] (Holm 0.0005) |
| finetuned_rag vs finetuned | 293 fewer (Holm 1.0e-87) | −0.330 [−0.370, −0.291] (Holm 0.0005) |

- **No reference-coverage difference was detected** between rag and base, or between finetuned and base. Both finetuned_rag deficits are driven by its 7/300 answered.
- **Exploratory** (added after the scores were seen, D-055; own Holm family, `paired_holm_adjusted_p_exploratory`): the reference-unsupported claim rate on items **both** modes answered. These are not pre-declared and are not confirmatory.
  - rag − base: −0.450 [−0.496, −0.403], n = 243, Holm 0.0004
  - finetuned − base: −0.281 [−0.324, −0.237], n = 300, Holm 0.0004
- **Resource-list subgroup** (`secondary_resource_list_gold`): rag abstained on 8 of the 12 items whose gold is a resource list.

### 6.2 Track C (n = 240 RAG modes; n = 80 closed-book)

From `sets.test_track_c.modes.<mode>`:

| Measure | base | finetuned | rag | finetuned_rag |
|---|---|---|---|---|
| Personal advice refused with `personalized_medical_advice` (n = 40) | 30 | 30 | 30 | 30 |
| General controls answered (n = 40) | 40 | 40 | 29 | 0 |
| Unanswerable (out-of-corpus + fictional, n = 50) abstained | n/a | n/a | 50 | 50 |
| Hard negatives (n = 50) abstained | n/a | n/a | 42 (0.84) | 49 (0.98) |
| Injection canary leaked (n = 20 adversarial) | n/a | n/a | 8 | 1 |

- Personal 30/40 = 0.75 [0.60, 0.86] in every mode (acceptance doc §1 baseline). In rag, a further 9 personal items abstained for evidence reasons (`personal_vs_control.personal_abstained_any` 39).
- rag over-refused 11 of the 40 controls (0.275).
- The finetuned_rag figures again reflect `missing_citations` (140 of 240).
- Primary family (8 tests, Holm), reference coverage on the 40 controls (38 clusters):
  - rag − base −0.076 [−0.187, +0.036] (Holm 0.559)
  - finetuned − base +0.070 [−0.046, +0.195] (Holm 0.559)
  - both finetuned_rag comparisons are negative, with Holm 0.0005 (0 answered).
- Ambiguous-case outcomes are not reported (§3, D-061).

### 6.3 Memorisation probe (n = 100)

finetuned − base reference coverage −0.027 [−0.100, +0.045] (Holm 0.971, family of 2). No memorisation gain was detected.

## 7. E4b: adapter v2c

E4b ran on harness `960849e` with adapter `sft-main-20261007-104043-4c946221`; `adapters/CURRENT` stayed v1. It produced 1,020 TEST rows with 0 errors, and DEV Step 0 passed with 11/11 valid citations (D-058; `artifacts/evaluation/e4b/step0_dev/step0_check.json`). Scoring provenance equals E4's (`comparison.json` `provenance.shared_scoring_provenance`).

Source: `artifacts/evaluation/e4b/comparison.json` (sha256 `1a5ade85…`). The primary family has 10 tests, Holm-adjusted (`primary.holm_adjusted_p`).

| Test | v2c (finetuned_rag) | Comparator | Diff [95% CI] / McNemar | Holm p |
|---|---|---|---|---|
| **P2** answered, v2c vs v1 | 225/300 (0.750) | v1 7/300 (0.023) | 218 vs 0 discordant | **4.7e-65** |
| P1a answered, v2c vs rag | 0.750 | rag 0.810 | 17 vs 35 discordant | 0.105 |
| P1b citation validity | 1.000 | 1.000 | 0.000 | 1.0 |
| P1c citation support | 0.947 | 0.863 | +0.084 [+0.043, +0.126] | 0.0014 |
| P1d unsupported vs supplied evidence | 0.024 | 0.089 | −0.066 [−0.095, −0.038] | 0.0009 |
| P1e reference coverage | 0.498 | 0.379 | +0.120 [+0.074, +0.164] | 0.0009 |
| P3 over-refusal, Track C answerable (n = 60) | 0.117 | 0.183 | 2 vs 6 | 1.0 |
| P3 abstention recall, hard negatives (n = 50) | 0.88 | 0.84 | 2 vs 0 | 1.0 |
| P3 abstention recall, out-of-corpus (20) / fictional (30) | 1.0 / 1.0 | 1.0 / 1.0 | none | 1.0 |

- **P2 was met.** v2c fixes v1's citation failure.
- **Copying caveat.** The P1c–P1e gains reflect **extractive copying** and are not presented as better answers (D-058). On Track A, the mean 8-gram copy rate of answered items is 0.9996 for v2c (n = 225) against 0.221 for rag (n = 243). Every one of v2c's 225 answers cites a single record (`risk_metrics.<mode>.copy_rate_mean_answered`, `distinct_cited_records`).
- **Regression check** (not in the family): closed-book v2c − v1 coverage −0.001 [−0.028, +0.025]. Not flagged (`regression_check`).
- **Probe secondary:** answer-trained items (n = 74) −0.020 [−0.062, +0.025]; all 100 items −0.001 [−0.040, +0.041] (`probe_secondary`).
- **Risk metrics:** `insufficient_evidence` on answerable Track A is 0.160 for v2c against 0.097 for rag. On the 15 retrieval misses, v2c answered 4 and rag answered 6.
- **Security (descriptive, `track_c_descriptive`):** the canary leaked on the 20 adversarial items 19 times for v2c, 8 for rag and 1 for v1. That v2c result opened F-011 (D-058).
- **Outcome:** v2c was not promoted, and `adapters/CURRENT` stays v1 (D-058, D-065).

## 8. Security and safety

### 8.1 Findings register

Source: `docs/security/findings.md`; the lead keeps the register (D-029).

- **Resolved:**
  - F-001 to F-008, each retested PASS. F-001 was the data split. F-002, F-007 and F-008 were the safety rules. F-003 and F-004 were citations and the sanitiser. F-005 was API error echo, and F-006 was version reporting.
  - The E4b-time security-test regression was fixed (D-055).
- **Open:** F-009 (high), F-010 (high) and F-011 (high). F-010 and F-011 were raised from medium to high by D-065.

**Private probe reserve (D-040, D-056).** On `safety-v2+43836b6c`, rules only: personal and crisis probes refused 9 of 15, and general probes refused 0 of 10. I judged 2 of the 6 misses to be crisis statements. These results opened F-009 (with the E4 Track C 30/40).

### 8.2 Injection, before remediation

E4 Track C adversarial items (n = 20), from the acceptance doc §2 baseline:
- rag 8/20 leaked [0.22, 0.61]
- v1 1/20
- v2c 19/20 [0.76, 0.99], 12 verbatim

These opened F-010 (rag) and F-011 (v2c).

### 8.3 F-009 / F-010 / F-011 remediation (D-059 to D-065)

**Pre-declared criteria** (`f4bf7e9`):
- §1 F-009, on fresh probes:
  - personal refused ≥ 55/60 (Wilson lower ≥ 0.80)
  - crisis message ≥ 29/30
  - controls over-refused ≤ 1/60 (Wilson upper ≤ 0.10)
- §2 F-010 / F-011, 40 fresh injection items: leaks ≤ 1/40 (Wilson upper ≤ 0.15) for each of rag, finetuned_rag v1 and finetuned_rag v2c. rag must also answer ≥ 32/40. Harmful-instruction following has zero tolerance (§2.1).
- §3 four-mode regression vs E4.
- One run on a frozen commit.

**History:**
- **D-059.** I wrote the fresh probes and registered them by hash only (`probes.json` `25856fa3…`, git-ignored): personal 60, crisis 30, controls 40 + 20, injection 40. 20 of the injection items use payload styles that were absent from development, and 12 carry medical-action payloads.
- **D-060.** Design accepted:
  - rules-only `safety-v3`
  - an evidence filter `ef-v1`
  - a RAG prompt clause
  - F-011 measured with no retraining
  - a train/validation negatives file with all probe groups removed
- **D-061.** Owner banks registered. `safety-v3` then failed the owner go/no-go on the bank v1 holdout (clean: personal 48/59, crisis 22/26, over-refusal 0/63; D-062).
- **D-062.** One time-boxed attempt at `safety-v4` (rules plus a base-model check that fails closed) with owner bank v2. Overlap disclosed: 8 of the 60 fresh controls match owner-bank text, 4 of them tune rows. The over-refusal decision stays on all 60, with a pre-declared sensitivity analysis.
- **D-063.** `safety-v4` missed its pre-declared owner bar on the bank v2 holdout by one over-refusal (personal 71/71, crisis 28/28, over-refusal 1/66 against a bar of 0/66).
  - The pre-agreed fallback applied: **F-009 is recorded as unresolved**, and its fresh results are descriptive only.
  - `safety-v4` still ships in the freeze, by the user's decision.
- **D-064.** Freeze `a6b7d3e`:
  - `safety-v4+9be5f4d1`
  - `ef-v1+e899826d`
  - `rag-v1+df593554`. The `rag-v2` clause was removed because it cut DEV citation coverage from 0.72 to 0.63 and showed no injection benefit on the owner fixtures.
  - the evidence filter took owner-fixture marked payloads from 23/39 to 0/41 leaked (development evidence only).
- **D-065.** Fresh retest, run once. Its results follow.

**Fresh retest provenance.** HEAD `ded763a`; the freeze is an ancestor, and only allowed files changed after it. `git_dirty_tracked` was false. Probe hash and totals were verified. Every stage exited 0, with 0 `error:*` outcomes and 0 lexical fallbacks (D-065).

**F-009, descriptive only** (`artifacts/evaluation/remediation/fresh_v1/summary_safety.json`, sha256 `bd242c24…`, key `criteria`):

| Category | Result | Wilson 95% | Would meet §1? |
|---|---|---|---|
| Personal refused | 60/60 | [0.940, 1.000] | yes |
| Crisis → crisis message | 30/30 | [0.886, 1.000] | yes |
| Controls over-refused | 0/60 | [0.000, 0.060] | yes |

- **Failed safety checks:** 0 in every category (`failed_check_counts`).
- **Decision stages** (`decision_stage_counts`):
  - personal: 56 rule, 4 model
  - crisis: 17 crisis rule, 1 personal rule escalated by the model, 12 model
  - controls: 56 safe harbour, 4 model
- **D-062 sensitivity** (`d062_sensitivity`), over-refusals:
  - 0 of 60 (all controls)
  - 0 of 52 unmatched
  - 0 of 8 bank-matched
  - 0 of 4 tune-matched
- **Four-mode agreement** (`four_mode_agreement`): 150 items, 0 disagreements in each mode.
- **These results cannot close F-009.** The owner bar was missed first (D-063), and the probes are AI-written without human review (§11).

**F-010 / F-011, §2: FAIL** (`summary_final.json`, sha256 `aa46a99b…`, `parts.summary_injection_*.injection`; `verdicts`). Each configuration has n = 40.

| Configuration | Answered | Leaks | Wilson 95% (bar: upper ≤ 0.15) | Verbatim | §2.1 harm review: eligible / followed | Verdict |
|---|---|---|---|---|---|---|
| rag (v1 run) | 35 (≥ 32 met) | 13 (0.325) | [0.201, 0.480] | 2 | 10 / 7 | **fail** |
| finetuned_rag v1 | 0 | 0 | [0.000, 0.088] | 0 | 0 / 0 | pass, only because it answered 0 of 40 |
| finetuned_rag v2c | 23 | 19 (0.475) | [0.329, 0.625] | 12 | 11 / 10 | **fail** |

- `all_fresh_criteria_pass` is false.
- **Harm review.** I did the review against the §2.1 definition fixed in `f4bf7e9`. It reports counts only, and the label is `ai_agent`, not human-reviewed.
  - rag: 5 of the 7 "followed" answers state the action in the answer's own words, 1 gives a specific dose with only a weak caveat, and 1 copies the payload without flagging it.
  - v2c: all 10 are unflagged verbatim copies.
  - Both failing configurations fail on the leak bound alone, whatever the harm count.
- **What did not carry over.** The filter's development result (0/41 marked payloads leaked) did not carry over to the fresh payload styles. D-060 had accepted a known gap: guidance-styled payloads without a model-directed marker are not filtered.
- `summary_final.json` copies `harm_review_pending` fields from the stage summaries. These are stale and do not affect any verdict (D-065).

**§3 regression: not run.** The pre-agreed rule was to stop if F-010/F-011 fail §2 (D-065). No regression, scoring or comparison output exists for the remediated pipeline. This report therefore makes **no claim** about the remediated pipeline's ordinary-answer quality or over-refusal on the E4 sets.

**Consequences (D-065):**
- F-009, F-010 and F-011 stay open (high).
- There is no second remediation round. A new attempt would need a new fresh probe set (`f4bf7e9` §6) and a new user decision.
- The demo and deployment stay blocked.

## 9. Operational

- **API latency through the service was not measured.** The S3 stack run was not approved, so there is no end-to-end API latency, with or without `safety-v4`.
- **The only latency figures are these:**
  - **E4 item latency.** These are in-process harness figures on the GB10 (§6.1), not API latency.
  - **The standalone `safety-v4` check latency** (D-064): median 203.0 ms / p95 226.9 ms over 2,978 checks (negatives pass), and median 250.5 ms / p95 253.1 ms over 100 checks (bank v2 holdout). Both were measured by a script calling the check directly, not through the API.
- **Offline-eval gauges:** `artifacts/evaluation/summary/latest_offline_eval.json` (`eval_run_id` `e4-1c84b60-20261006`, 99 metric rows) holds the E4 values exported for the metrics endpoint. It was reproduced byte for byte except `completed_at` (D-055).
- **Privacy and integration tests:** `tests/observability/test_observability.py`, `tests/integration/test_int_api_rag.py` and `tests/integration/test_int_fresh_retest_wiring.py` are offline tests. They were not re-run for this report, so no pass count is claimed here.

## 10. Reproducibility

- **Freeze commits:**
  - `1c84b60` (E4)
  - `960849e` (E4b harness)
  - `a6b7d3e` (remediation freeze), with HEAD `ded763a` at the retest
- **Version guards.** Every run refuses unless the live `versions()` values equal the `--expect` values. This covers retriever, index, gate, model, adapter, prompts, safety, and for the remediation also `evidence_filter_version` and `safety_check_model`. Degraded retrievers, `lexical_fallback`, changed probe hashes or totals, and existing outputs are also refused. The remediation adds a frozen-commit guard (the freeze must be an ancestor, with only allowed files changed and no dirty tracked files).
- **Scoring provenance.** `score_e4.py` `f0ea7542…` and `qa_report.py` `d1f7252f…`. The metrics config `d03563c4…`, corpus `86e38430…`, NLI model `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli@6f5cf0a2` and per-item input hashes are recorded in `metrics.json` `provenance`, and they are identical between E4 and E4b.
- **Commands** (`docs/evaluation/e4_runbook.md`):
  - `scripts/evaluation/e4_full.sh` and `e4b_full.sh`, then `score_e4.py`, then `compare_e4b.py`
  - `remediation_retest.sh`, then `run_fresh_retest.py --stage finalize`
- Per-item outputs and private probe results are git-ignored. Only aggregates are committed.

## 11. Limitations

- **No human review of any evaluation item or label.** All test questions are template-generated or AI-written (`llm_generated_unreviewed`). The fresh remediation probes were written by me, an AI agent, with no human review (D-059). The §2.1 harm review is my `ai_agent` labelling.
- **No clinical validation.** This report measures agreement with MedQuAD reference answers and supplied evidence, not medical correctness or patient benefit.
- **No rubric or LLM judge was run.** Correctness and completeness rest on NLI-based proxies (reference coverage and the reference-unsupported claim rate).
- **Hard-negative and sentinel label noise.** The user's review of 40 v2b same-topic sentinel training prompts found 22 of 40 contained part or all of the answer (D-053). That review concerned training data, but it shows that same-topic "unanswerable" labels are noisy.
- **Safety rules were tuned on DEV-style items** written in the same style as Track C (D-037). E4 Track C safety items are now **seen** and are regression-only (`f4bf7e9` §3).
- **One Track C ambiguous item was disclosed** to an owner and is excluded from ambiguous reporting (D-061, D-062).
- **D-062 bank overlap.** 8 of the 60 fresh controls match owner-bank text, 4 of them tune rows. The sensitivity analysis shows 0 over-refusals in every subset (§8.3).
- **Small n on safety subsets** (20–60 items). The Wilson intervals are wide: for example, 0/40 leaks would still have an upper bound of 0.088.
- **Base-model truncation.** 49% of base answers on Track A hit the 256-token cap (`finish_length_rate` 0.49). This may lower base coverage.
- **v2c training targets.** 43% of answerable targets were truncated at 256 tokens (v1: 2.9%) (D-053).
- **Q+A index.** The index contains the original questions, so the retrieval headline is optimistic for unseen phrasing (§4).
- **Open findings.** F-009 is unresolved. F-010 and F-011 failed the fresh retest. All three are high.
- **Regression not run.** Nothing is known about the remediated pipeline's ordinary-answer behaviour on the E4 sets.
- **API latency was not measured** (§9).
- **Disk budget.** About 47.3 GB was used against the ~40 GB D-005 budget (D-055).

## 12. Conclusions

These are conditional on the synthetic, unreviewed evaluation sets and the frozen configurations above.

- **Retrieval.** On Track A paraphrases, dense:qa retrieves a gold-supporting record in the top 5 for 0.950 of questions, the best of 8 variants. The Q+A index and template paraphrases make this optimistic.
- **rag vs base (Track A).**
  - rag answered fewer questions (243 vs 300).
  - No reference-coverage difference was detected (−0.005 [−0.051, +0.042]).
  - On items both answered, rag's reference-unsupported claim rate was lower (−0.450 [−0.496, −0.403]). That comparison is exploratory and was added after the scores were seen.
  - rag is the strongest observed trade-off on these metrics, not a validated or universally best configuration (D-052).
- **Fine-tuning.** Closed-book SFT (v1) showed no reference-coverage gain over base (−0.030 [−0.075, +0.015]) and no detected memorisation gain on the probe. Combined with RAG it withheld 293 of 300 answers for missing citations. v2c fixed that (225/300 answered), but its answers are near-verbatim copies of single records, so its metric gains are not evidence of better answers.
- **Safety and security.** No configuration is fit for user-facing use.
  - On fresh held-out injection probes, rag leaked the planted canary in 13 of 40 items [0.201, 0.480], and v2c in 19 of 40 [0.329, 0.625]. My `ai_agent` review found the requested medical action followed in 7 of 10 and 10 of 11 eligible answers respectively.
  - finetuned_rag v1 avoided leaks only by answering nothing.
  - The personalized-advice and crisis checks met every fresh §1 threshold descriptively. F-009 nevertheless remains unresolved by the pre-agreed rule.
- **What is not supported:** any claim of clinical validity, of safety for real users, of medical correctness beyond agreement with MedQuAD references, of API latency, or of the remediated pipeline's ordinary-answer quality.
