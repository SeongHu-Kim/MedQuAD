# Model card: answerability classifiers (non-clinical)

**Task.** Given a question and one evidence passage, predict whether the passage is adequate to answer the
question. This is a supervised NLP task over synthetic labels. It is **not** diagnostic, prognostic or medical
confidence, and it has not been clinically validated. In the RAG pipeline the score gates whether the generator answers
or abstains (`AnswerabilityPredictor`, aggregate = max over evidence passages).

All numbers below come from `artifacts/models/answerability/{lexical_lr,keras_bigru}/metrics.json` and
`comparison.json`, produced by `scripts/training/train_answerability.py` (stages `pairs`, `baseline`, `keras`,
`scores`, `compare`).

## Data and labels

| Item | Value |
|---|---|
| Source | Frozen MedQuAD split `split-20261006-2f0fb25ee6d8`, corpus `medquad-1.0.0-86e384302357` (sha256s in `pairs/pairs_stats.json`) |
| Pair rules | `ans-pairs-v1` (`src/medquad_qa/training/answerability_pairs.py`), agreed with evaluation-safety-engineer (D-033) |
| Label provenance | `synthetic_rule` only. No human labels. |
| Positive | The question's own answer (`own_answer=true`), plus at most one other record with the same topic AND question type |
| Negatives (≈1:1) | `easy_random` (different topic and group), `same_topic_diff_qtype` (same disease, different aspect), `lexical_hard_bm25` (top BM25 hit from the same split with a different topic, duplicate group and split group) |
| Controls | C1 different split group for easy/BM25 negatives; C2 negatives textually equal to a positive are dropped (4 in train, 0 elsewhere); C3 boilerplate answers excluded everywhere; C4 own-answer flag |
| Evidence text | First ≤200 words of the answer |
| Pairs | train 30,506 / validation 3,960 / test 3,876 (built within each split; IDs-only manifests `pairs/pairs_manifest_*.jsonl`) |

Known label noise:
- A `same_topic_diff_qtype` negative can be partly answerable. For example, an overview answer often mentions symptoms.
- Using only the first chunk can miss the answer span.
- "Same topic + type" is a proxy for adequacy, not a semantic judgement.

## Models

| | Lexical LR baseline | Keras BiGRU |
|---|---|---|
| Version | `answerability-lexlr@3defcd00a31f` | `answerability-keras-bigru@0897a41639ce` |
| Inputs | 8 overlap features (coverage, IDF coverage, Jaccard, bigram coverage, TF-IDF cosine, lead-30 coverage, log lengths); IDF fitted on train evidence | Shared embedding (128) + BiGRU (64) siamese encoder over question (32 tokens) and evidence (256 tokens); [u, v, u−v, u⊙v] plus the 8 lexical features → dense 64 → sigmoid |
| Training | scikit-learn LogisticRegression (C=1, standardised features, seed 42) | TF 2.21 / Keras 3.15, CPU only, Adam 1e-3, batch 128, ≤8 epochs, early stopping on validation AUC (patience 2; 4 epochs run, epoch-2 weights kept), 3.42 M params, seed 42 with deterministic ops |
| Threshold (max-F1 on validation, D-018) | 0.3072 | 0.4892 |
| Serving | **Yes, frozen gate for E4.** JSON + numpy only (`models.load_answerability_predictor`). No pickle, sklearn or TF at runtime. | Not in the frozen config. TF must not run in the API process (D-036; also an import-order segfault with Triton). D-014 does allow an ONNX path (tf2onnx → onnxruntime). |

## Results (test pairs, frozen validation threshold; scored once)

The headline excludes own-answer positives (evaluator condition C4). Those are lexically near-trivial. Positive
prevalence in the headline set is about 16%. ECE uses 10 equal-width bins (`medquad_qa.evaluation.classifier_metrics`).

| Model | ROC-AUC | PR-AUC | Brier | ECE | Precision / Recall / F1 at threshold |
|---|---|---|---|---|---|
| LR, excl. own answer | 0.784 | 0.377 | 0.165 | 0.213 | 0.268 / 0.810 / 0.403 |
| Keras, excl. own answer | 0.934 | 0.786 | 0.086 | 0.077 | 0.550 / 0.791 / 0.649 |
| LR, all test pairs | 0.826 | 0.807 | 0.170 | 0.038 | 0.702 / 0.876 / 0.779 |
| Keras, all test pairs | 0.961 | 0.961 | 0.081 | 0.038 | 0.891 / 0.891 / 0.891 |

Share of each negative type predicted "answerable" at the threshold (test):

| Model | easy_random | lexical_hard_bm25 | same_topic_diff_qtype |
|---|---|---|---|
| LR | 1.0% | 35.4% | **84.9%** |
| Keras | 3.7% | 16.3% | 13.7% |

ROC-AUC by negative type (test, excl. own answer; all positives vs that negative type), computed from
`scores_test.jsonl`. evaluation-safety-engineer's independent recomputation (`artifacts/evaluation/classifier/metrics.json`)
matches the headline table exactly and reports the same per-type values (LR 0.944 / 0.825 / 0.536; Keras 0.914 on
same_topic_diff_qtype).

| Model | easy_random | lexical_hard_bm25 | same_topic_diff_qtype |
|---|---|---|---|
| LR | 0.944 | 0.824 | **0.536 (near chance)** |
| Keras | 0.967 | 0.914 | 0.914 |

Interpretation:
- The lexical baseline mostly detects *topical* match. It cannot tell "symptoms of X" evidence from "treatment of X" evidence.
- The BiGRU learns the question-aspect interaction and is much stronger on these synthetic labels.
- Calibration of both models was fitted on a mix that includes own-answer positives. ECE is therefore worse on the headline subset, especially for LR.

## Limitations and intended use

- These are synthetic labels from dataset structure, not human adequacy judgements. The test metrics measure agreement
  with the pair rules, not real-world answerability.
- The validation split chose the threshold and, for Keras, the early-stopping epoch. Test was used only for reporting.
- **What the metrics measure.** All labels are synthetic (pair rules). These metrics measure how well each model
  *recovers the rules*, not real-world answerability.
- The serving gate uses the weaker LR model. Swapping it now would reopen the frozen E4 configuration, so it stays
  frozen for E4. Expect it to pass same-disease, wrong-aspect evidence. The RAG pipeline's citation checks and the
  `INSUFFICIENT_EVIDENCE` sentinel remain the other abstention layers.
- **Recommended future gate:** the Keras BiGRU exported with tf2onnx and served with onnxruntime (D-014). This keeps TF
  out of the API process. On these labels it would raise headline ROC-AUC from 0.784 to 0.934 and cut wrong-aspect
  acceptance from 84.9% to 13.7%. The ONNX export has not been built or validated yet.
- Not for clinical decisions. The scores are not probabilities of medical correctness.

## Reproduce

```bash
.venv/bin/python scripts/training/train_answerability.py --stage pairs      # verifies export sha256s
.venv/bin/python scripts/training/train_answerability.py --stage baseline
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/training/train_answerability.py --stage keras   # separate process
.venv/bin/python scripts/training/train_answerability.py --stage scores     # IDs-only per-pair scores
.venv/bin/python scripts/training/train_answerability.py --stage compare
```
