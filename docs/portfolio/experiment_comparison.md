# Experiment comparison report

Owner: lead. A summary of the controlled experiments. The full analysis, with every source path and key, is the independent evaluation report [`docs/evaluation/e6_report.md`](../evaluation/e6_report.md) (E6, written by evaluation-safety-engineer). Every number below is copied from E6 or a decision row. Nothing here is clinically validated, and every evaluation question and label is synthetic: template-generated or AI-written without human review.

`[a, b]` is a 95% interval: Wilson for single proportions, paired cluster bootstrap (10,000 resamples, clusters = split group) for differences. "Holm" is the Holm-adjusted p within the pre-declared family.

## 1. Retrieval (Track A, n = 300 paraphrased questions)

Source: `artifacts/evaluation/retrieval/scores_test_track_a.json` (E6 §4).

| Variant | R@1 | R@5 | MRR@20 |
|---|---|---|---|
| bm25:answer | 0.503 | 0.743 | 0.604 |
| bm25:qa | 0.607 | 0.843 | 0.707 |
| dense:answer | 0.673 | 0.890 | 0.765 |
| **dense:qa (frozen)** | **0.830** | **0.950** | **0.884** |
| hybrid_rrf:qa | 0.747 | 0.920 | 0.823 |
| hybrid_rrf+ce:qa | 0.693 | 0.870 | 0.771 |

- dense:qa beat dense:answer by +0.060 [+0.033, +0.087] R@5, and neither hybrid nor the cross-encoder improved on it.
- **Caveat:** the "qa" index contains the original MedQuAD questions, and Track A questions are paraphrases of them, so the headline is optimistic. On the 100 AI-written questions dense:qa R@5 is 0.92, against 0.965 on the 200 template questions.

## 2. Answerability classifier (TensorFlow baseline)

Source: `artifacts/evaluation/classifier/metrics.json` (E6 §5); test set 3,876 pairs.

| Model | AUROC | AUROC excl. own answer | AUROC same topic, different question type |
|---|---|---|---|
| Lexical LR (served gate) | 0.826 | 0.784 | **0.536** |
| Keras BiGRU (not served) | 0.961 | 0.934 | 0.914 |

The served gate is near chance on same-topic negatives. On E4 Track C, 49 of 50 hard negatives passed the gate, and the generator's INSUFFICIENT_EVIDENCE sentinel caught 41 of them. The Keras model was more accurate but was not served (D-046).

## 3. Four-mode comparison, E4 (adapter v1; freeze `1c84b60`)

Source: `artifacts/evaluation/e4/metrics.json` (E6 §6). Track A, n = 300 answerable questions.

| Mode | Answered | Reference coverage (abstained = 0) | Citation validity / support |
|---|---|---|---|
| base | 300 | 0.384 | n/a |
| rag | 243 | 0.379 | 1.000 / 0.857 |
| finetuned (v1) | 300 | 0.354 | n/a |
| finetuned_rag (v1) | 7 | 0.023 | 1.000 / 1.000 (n = 7) |

Pre-declared primary family (Holm):
- rag vs base: reference coverage −0.005 [−0.051, +0.042] (Holm 1.0). **No difference detected.** rag answered 57 fewer questions.
- finetuned vs base: −0.030 [−0.075, +0.015] (Holm 0.605). **No fine-tuning gain detected.**
- finetuned_rag: answered 7/300, because the closed-book v1 adapter emits no `[E#]` citation labels (found on DEV before any TEST output, D-051).
- Memorisation probe (n = 100): finetuned − base −0.027 [−0.100, +0.045]. No memorisation gain detected.
- Exploratory, added after the scores were seen (D-055): on items both answered, rag's reference-unsupported claim rate was lower than base's by −0.450 [−0.496, −0.403]. Not confirmatory.

## 4. Adapter v2c, E4b (RAG-formatted fine-tuning)

Source: `artifacts/evaluation/e4b/comparison.json` (E6 §7). Primary family of 10 tests, Holm.

- **P2 met:** finetuned_rag answered 225/300 with v2c against 7/300 with v1 (Holm 4.7e-65).
- P1a: v2c answered 0.750 against rag 0.810, not significant (Holm 0.105).
- P1c–P1e (citation support +0.084, unsupported-vs-evidence −0.066, reference coverage +0.120) were Holm-significant, **but they reflect extractive copying**: the 8-gram copy rate is 0.9996 for v2c against 0.221 for rag, and every v2c answer cites a single record. They are not presented as better answers (D-058).
- v2c was not promoted; `adapters/CURRENT` stays v1.

## 5. Remediation retest (D-065)

See the [safety and security report](safety_security_report.md). In short: the injection fixes failed on fresh held-out payloads (rag 13/40 leaked, v2c 19/40), so the §3 regression comparison was not run, and no claim is made about the remediated pipeline's ordinary-answer quality.

## What these experiments support

- A dense question+answer index gave the best retrieval on paraphrased questions, with the optimism caveat above.
- RAG did not change reference coverage relative to the base model but abstained more; its lower unsupported-claim rate is exploratory.
- Closed-book LoRA fine-tuning showed no measurable gain; RAG-formatted fine-tuning fixed citation output but learned to copy.
- None of this establishes clinical validity or medical correctness beyond agreement with MedQuAD reference answers.
