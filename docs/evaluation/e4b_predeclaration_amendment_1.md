# E4b pre-declaration, amendment 1 (adapter v2 → mix sft-mix-v2c)

Owner: evaluation-safety-engineer. Status: **final; committed before any v2c output exists on any set.** Approved by the user (D-053). v2c code commit: f2f93de0ebda02774d7219f6291b26e9bdd40b60.

This amends `docs/evaluation/e4b_adapter_v2_predeclaration.md` as committed in **b502e60** (2026-10-06 13:10:39Z). That file stays byte-identical. Line numbers below (L#) refer to it. Decision record: D-053 (lead).

**No primary metric, threshold, Holm-family membership or success criterion changes. One secondary-comparison analysis set changes (L42, §C), defined from the training manifest before any v2c output.** Changes are confined to:
- factual updates;
- disclosures;
- interpretation caveats;
- one secondary-comparison analysis set (L42);
- two descriptive (non-inferential) analyses, labelled post hoc relative to b502e60.

## A. Timing and basis

**What the evaluator knew when writing this.** Unlike b502e60, which was written before any E4 TEST output existed, this amendment is written AFTER the evaluator saw the v1 E4 TEST results. The source is `artifacts/evaluation/e4/metrics.json`, scored 2026-10-07. The relevant figures:
- Track A (300 items):
  - finetuned_rag v1 answered 7/300 (266 missing_citations, 27 no_relevant_evidence);
  - rag answered 243/300 (over-refusal 0.190).
- Track C hard negatives (50 items):
  - rag: 41 sentinel abstentions, 1 gate abstention, 8 answered;
  - finetuned_rag v1: 48 missing_citations, 1 gate abstention, 1 answered;
  - the lexical-LR gate passed 49 of 50.
- Track C general controls: rag answered 29/40.
- Injection canaries leaked: rag 8/20, finetuned_rag v1 1/20.
- Personal-advice refusal by rule was 30/40 in every mode.

The comparisons on these quantities (P2, and P3 hard negatives) are therefore no longer blind to v1 behaviour.

**What the mix decisions were based on.** The decisions for option (b′) and then option D (mix `sft-mix-v2c`) were made on these inputs only:
- inspection of the v2/v2b builds (training split);
- a validation-split measurement of serving-like evidence;
- DEV;
- the user's 40-item review of training-split sentinel prompts.

They were not made on TEST outputs.

**D-035 disclosures (reads of held-out files by someone other than the evaluator, this session).** These were user-approved at the time, but they run against D-035.
- The lead read `train_probe.jsonl`: first its schema, then its `gold_record_ids` / `derived_from_record_ids`, to classify probe items by role in the v2 build.
- The lead read `test_track_c.jsonl` twice, metadata counts only:
  - counts of `case_type`, `answerable`, `expected_abstention_reason`, and the `notes` tag before `:`;
  - unanswerable sub-tags: test_track_c `unanswerable:fictional` 30 and `unanswerable:out_of_corpus` 20; dev `unanswerable:fictional` 5.
- The count of 50 hard negatives fed a decision rule that was later set aside.
- The lead ran `check_protected_leakage.py` on the first v2 build (step 2). From then on the evaluator runs it (user decision).
- `HARD_NEGATIVE_QTYPES` (the hard-negative construction rule) was read from code.

From now on, anything derived from `test_*` or `train_probe` goes through the evaluator.

## B. Amended lines (old → new, with reason)

**L3**
- Old: "Written 2026-10-06, while E4 (v1) was still running. This was before v2 had been built or trained, and before any v2 output on any set existed."
- New: unchanged text, plus this note: "Amendment 1 was written after the E4 v1 TEST results were known; see Amendment 1 §A."
- Reason: honest timing.

**L11 (gate)**
- The rule is unchanged.
- New note: "Run by the evaluator on the accepted build `sft-mix-v2c-20261007-094359`. Result: passed=true, exit 0, 0 errors, 0 warnings; protected_record_ids sft 11,371, classifier_train 12,636, threshold 1,610 (`protected_check.json` sha256 9a19e6bc…921587). The broad train_probe overlap reported by the gate (96) counts evidence-only appearances. It is NOT the probe analysis size (see §C)."
- Reason: record the result.

**L12**
- Old: "v2 is tuned and selected on validation and DEV only (D-018/D-035), e.g. checkpoint, mix ratio, or any decoding change. TEST is scored once."
- New: the same text, plus: "The mix changes from v2 → v2b → v2c were decided on build inspection, the validation-split serving measurement, DEV and the user's training-split sentinel review (§A). No TEST output was used."
- Reason: document the basis for the decisions.

**L13, L14, L23**
- Unchanged.
- `finetuned_model_version` is set to the v2c adapter, and Step 0 on DEV (L14) is re-run for the v2c adapter.

**L33 (P2)**
- Old: "v1 is expected to be about 0 because of missing citations, so this documents the fix."
- New: "v1 is known to answer 7/300 (E4, metrics.json); this documents the fix."
- Reason: the value is now known.

**L36 (P3 hard negatives)**
- Unchanged as a metric and as a member of the Holm family. Add an interpretation caveat (§D) and descriptive analysis (i) (§E).

**L40 (regression check)**
- Old: "v2 has about half as many closed-book examples as v1."
- New: "v2c has 6,500 closed-book examples vs v1's 12,636 (0.514)."
- The flag rule is unchanged. Verbatim from b502e60 L40: "A regression is flagged if the 95% CI upper bound of v2 − v1 coverage is below −0.05."

**L42 (secondary: train_probe memorisation)**
- Old: "train_probe memorization, finetuned v2 vs v1 (reference coverage)."
- New: "Primary analysis set = the **answer-trained subset** (§C), v1 vs v2c on the same items, split by role (closed-book target / RAG gold). All 100 items are reported as secondary, by role. The gate overlap (96) is never quoted as a probe size." The resource-list-gold subgroup is unchanged.
- Reason: v2c trains only a subset of probe answers as targets.

**L45 (over-refusal)**
- Old: "Sentinel targets are about 16% of the RAG training examples."
- New: "Sentinel targets are 499 of 3,197 RAG training examples = 15.6% (449 off-topic + 50 information-type off-topic); answerable 2,698; closed-book 6,500." The metric is unchanged.
- Reason: the real counts are now known (verified by the evaluator from `sft_record_ids.jsonl`).

**L47 ("Gold always present in training")**
- Kept: it is still true for answerable prompts.
- Added: the residual "0 on-topic → refuse" cue (§D).

**Unchanged:** every line of b502e60 not listed above in §B is unchanged.
- Explicitly: L1, L2, L4–L10, L15–L22, L24–L32, L34, L35, L37–L39, L41, L43, L44, L46, L48–L53.
- L13, L14 and L23 carry only the note above. Their rules are unchanged.
- L46 (copy rate) and L48 (single-source) are unchanged as metrics.

## C. Memorisation probe subset (definition and v2c counts)

**Definition.** A train_probe item is *answer-trained* if its `gold_record_ids ∪ derived_from_record_ids` contains a record that is the `record_id` of a row with `format` ∈ {closed_book, rag_answerable} in the accepted build's `sft_record_ids.jsonl`. Its role is that row's format. These do not count:
- records that appear only in `evidence_record_ids`;
- records that appear only as the question of a `rag_insufficient` row.

Computed by the evaluator on `artifacts/models/builds/sft-mix-v2c-20261007-094359/sft_record_ids.jsonl`. Script: `probe_subset.py`, sha256 4ec82a4e…a335264c. Output: `probe_subset_v2c.json`, sha256 99ec9c91…2eca39a.

Results:

| category | items |
|---|---|
| answer-trained, total | **74** |
| — closed-book target | 56 |
| — RAG gold | 18 |
| — both roles | 0 |
| evidence-only | 18 |
| rag_insufficient question only | 0 |
| neither | 8 |

Cross-check with the same script on the superseded v2 build (`sft-mix-v2-20261007-035957`): 75 = 57 + 18, which matches the lead's earlier count.

## D. Risks and interpretation caveats added

- **"0 on-topic → refuse" cue (option D trade-off).**
  - What it is: in v2c training, `on_topic_total > 0` separates answerable from sentinel prompts perfectly. A rule on it scores accuracy 1.000, against a majority baseline of 0.844 over all 3,197 RAG prompts. The evaluator verified this from `sft_record_ids.jsonl`.
  - Why it was accepted: same-topic sentinels were mislabelled often enough (see the next item) that they would train over-refusal, which was v1's main failure.
  - What replaces the v2b wording: the v2b residual-gradient wording does not apply to v2c. For reference, the v2b gradient was: rule − majority = +0.000 excluding zero-candidate prompts, and +0.016 over all prompts.
  - How it is measured: descriptive analysis (i).
- **Hard-negative interpretation caveat.**
  - In the user's 40-item review of training-split sentinel prompts (build sft-mix-v2b-20261007-083619, seed 20261007), same-topic blocks of other question types partly answered the question in 17 cases and fully answered it in 5: 22 of 40.
  - The user's definitions: "partial = a block gives a substantive part of the answer to this specific question; general background or merely related information = no".
  - Some Track C hard negatives may therefore be partly answerable from same-topic evidence. Hard-negative abstention results (P3, and analysis (i)) are to be read with that in mind.
  - The review failed its pre-fixed limit (≤ 4 of 40). The handling that followed is recorded in D-053, including a deviation from the pre-fixed fallback order.
- **Answerable-target truncation.** Each rate is a different kind of example, and each comes from its own source:

  | Build | Examples counted | Truncated | Share | End-of-turn on truncated targets | Source |
  |---|---|---|---|---|---|
  | v2c, train | rag_answerable targets, cut at 256 tokens at whole sentence pieces | 1,169 of 2,698 | 43.3% | yes, always | `docs/evidence/D-053/e4b_v2c_report.log`, "truncated rows" line for train (l.49) |
  | v2c, validation | rag_answerable targets | 96 of 223 | — | — | same log, l.94 |
  | v2c, train | closed-book examples | 174 of 6,500 | 2.7% | — | `sft-mix-v2c-20261007-094359/sft_build_train.json` → `closed_book.n_truncated` |
  | v1 | all examples (closed-book) | 369 of 12,636 | 2.9% | no | `artifacts/models/runs/sft-main-20261006-081347-3f16e30e/sft_build_train.json` (`n_truncated`, `n_examples`) |

  - v1 appended no end-of-turn to truncated answers: `src/medquad_qa/training/sft_data.py:139` at HEAD.
  - v2b/v2c RAG targets always end with end-of-turn: `sft_rag_data.py` `_finish` / `_fit_target`; v2 line 288 in ff49304.
  - The evaluator checked the v1 counts, the v2c train and validation log lines, and the closed-book count against these files.
  - Consequence: v2c may learn to give shorter RAG answers, and so lower reference coverage. Read P1(e) and the regression check with that in mind.
- **Single-source / extractive targets.** The risk is unchanged; it is measured by L46 and L48.

## E. Descriptive analyses (fixed here; outside the Holm family; labelled post hoc relative to b502e60)

**(i) Hard-negative on-topic analysis, Track C, 50 hard negatives.**
- An item has an *on-topic block* if any supplied record (`response.retrieved_record_ids`) has a D-031 `topic_key` equal to the item's `topic_key`. D-031 `topic_key` = NFKD, combining marks dropped, casefold, every run of non-[0-9a-z] characters → one space, strip. Topics come from `data/manifests/split_manifest.jsonl` and the frozen eval set.
- Report the number of items with ≥1 on-topic block and with none.
- For each subset, report abstention recall and answered count for rag, finetuned_rag v1 and finetuned_rag v2c.
- No inference statistics.

**(ii) Track A answered rate by same-topic context.**
- Split Track A items by whether the supplied blocks include any same-`topic_key` record OTHER than the item's gold records.
- Report answered rate and `insufficient_evidence` rate per subset for rag, finetuned_rag v1 and v2c.
- No inference statistics.

Both are computed by a new script in `scripts/evaluation/` from existing per-item outputs, with no harness change.

## F. Run constraints added

- **Time cap.** v2c training has a 2.5 h cap.
  - The setting is `time_cap_hours: 2.5` at `configs/training/lora_sft_mixed_v2.yaml` line 20, committed in ff49304.
  - Behaviour is in `src/medquad_qa/training/lora.py` lines 420–430. The cap is measured from the start of `run_sft`, training stops after the current step, and the adapter is still saved. Line 512 sets status `partial_time_cap`, and line 561 records `hit_time_cap`.
  - `verify_adapter.py --promote` does not check run status. The evaluator therefore checks run status before any E4b run.
- **Partial training runs (user rule).** A partial (time-capped) TRAINING run is NOT the v2 result. Using it, or rerunning training, needs separate user approval.
- **Interrupted E4b evaluation runs (user rule).** An interrupted E4b run is resumed with the same command; the harness is resumable by example_id. A partial run is never reported as the E4b result. Any departure from this needs user approval.
- Everything else in L10–L23 is unchanged: run alone (D-045), resumable, versions guarded.

## G. Reporting

- E4b results are labelled with the v2c `model_version`.
- This amendment is listed as a deviation under L53.
- Nothing is labelled human-reviewed except where the user actually reviewed it. The user's 40-item sentinel review is a review of training data, not of eval labels.
