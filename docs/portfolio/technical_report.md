# Evidence-grounded medical question answering on MedQuAD: a controlled comparison of retrieval, fine-tuning and safety controls

Technical report (portfolio). **Not a published or peer-reviewed paper. Not clinically validated.** Owner: lead. Every number is taken from the independent evaluation report ([`docs/evaluation/e6_report.md`](../evaluation/e6_report.md), "E6") or a decision row (D-xxx in [`docs/decisions.md`](../decisions.md)).

## Abstract

We built a non-clinical research prototype that answers general medical-information questions from the MedQuAD corpus with citations to retrieved records, abstains when evidence is inadequate, and refuses personal medical advice. We compared four configurations under frozen, pre-declared evaluations: a base 4B instruction model, retrieval-augmented generation (RAG), a LoRA fine-tuned model, and fine-tuned RAG. On 300 paraphrased answerable questions, RAG showed no detected difference from the base model in reference coverage (−0.005 [−0.051, +0.042]) while abstaining more, and closed-book fine-tuning showed no gain (−0.030 [−0.075, +0.015]). A second, citation-formatted adapter fixed the fine-tuned RAG citation failure (225/300 vs 7/300 answered) but learned to copy single records extractively. A pre-declared remediation of safety and evidence-injection findings failed its fresh held-out retest for injection (13/40 and 19/40 canary leaks against a bar of ≤ 1/40). All evaluation items are synthetic and not human-reviewed.

## 1. Task and data

- **Corpus.** MedQuAD (NIH sources) from a local Kaggle CSV: 16,412 rows → 16,336 records after 76 documented exclusions (E6 §2). The export has no source URLs or document IDs, so citations are record-level.
- **Splits.** Group-aware, deterministic: 13,014 / 1,665 / 1,657 records in 3,338 / 428 / 446 groups (`split-20261006-2f0fb25ee6d8`, D-034). Groups are built from content and topic; an initial leakage verification failed (F-001) and the fixed rules passed a retest (E6 §2).
- **Evaluation sets** (frozen, sha256-pinned): DEV 60 (the only tuning set), Track A 300 answerable paraphrases, Track C 240 robustness items (controls, personal advice, out-of-corpus, fictional, hard negatives, injection, ambiguous, conflicting), and a 100-item memorisation probe (E6 §3). Questions are template-generated or AI-written; no label is human-reviewed.

## 2. System

Retrieval uses a dense index (BAAI/bge-small-en-v1.5 in embedded Qdrant) over question+answer text, chosen on DEV (D-037). A langchain-core pipeline applies safety rules in every mode, retrieves the top 5 records, filters instruction-like evidence sentences (remediation only), gates on a lexical answerability classifier, builds a prompt with `[E#]` evidence blocks, generates with Qwen3-4B-Instruct-2507, and validates that every citation refers to supplied evidence. See [architecture.md](architecture.md).

## 3. Methods

- **Modes:** base, rag, finetuned (closed-book LoRA, v1) and finetuned_rag (D-023).
- **Metrics:** reference coverage and reference-unsupported claim rate by NLI against MedQuAD answers; citation validity and support; abstention and over-refusal; retrieval recall and MRR (`docs/evaluation/metrics.md`).
- **Statistics:** paired cluster bootstrap (10,000 resamples, clusters = split group) for differences; McNemar for answered rates; Holm adjustment within pre-declared families (E4: 8/8/2; E4b: 10); Wilson intervals for single proportions.
- **Controls:** frozen commits and version guards on every run; leakage gates against the training data (E6 §2, §10).

## 4. Results

**Retrieval** (Track A): dense:qa R@5 0.950, best of 8 variants; hybrid and cross-encoder variants did not improve on it. The index contains the original questions, so the headline is optimistic; on AI-written questions R@5 was 0.92 (E6 §4).

**Answerability:** a Keras BiGRU (AUROC 0.961) outperformed the served logistic-regression gate (0.826), which was near chance on same-topic negatives (0.536). In the pipeline the generator's sentinel caught most hard negatives that passed the gate (41 of 49) (E6 §5).

**Four-mode comparison** (E4, Track A): answered 300 / 243 / 300 / 7 and reference coverage 0.384 / 0.379 / 0.354 / 0.023 for base / rag / finetuned / finetuned_rag. No reference-coverage difference was detected for rag vs base or finetuned vs base. finetuned_rag failed because the closed-book adapter emits no citation labels, which DEV had shown before any test output (D-051). Exploratory and post hoc: on items both answered, rag's unsupported-claim rate was lower than base's (−0.450 [−0.496, −0.403]) (E6 §6).

**Citation-formatted adapter** (E4b): v2c answered 225/300 in finetuned_rag. Its citation-support and coverage gains were Holm-significant but reflect extractive copying (8-gram copy rate 0.9996 vs 0.221 for rag; every answer cites one record) (E6 §7, D-058).

**Remediation** (D-059 to D-065): criteria and fresh held-out probes were registered before any remediation code. The personal-advice/crisis remediation missed its owner bar by one over-refusal (D-063); its fresh results (60/60, 30/30, 0/60) are descriptive only. The injection remediation failed: rag leaked 13/40 [0.201, 0.480] and v2c 19/40 [0.329, 0.625]; an AI-agent harm review found the requested medical action followed in 7/10 and 10/11 eligible answers. The regression check was not run, by the pre-agreed rule (E6 §8).

## 5. Discussion

- Retrieval quality was high but optimistic for unseen phrasing; most of RAG's value here was in abstention and citation grounding, not in reference coverage.
- Fine-tuning a 4B model on MedQuAD answers did not add measurable reference coverage. Teaching it to cite produced copying, which inflates overlap-based metrics; copy rate should be reported next to any such gain.
- Development fixtures written by the system's own authors were a poor proxy for fresh adversarial payloads. Registering held-out probes by hash before development is what made this failure visible.

## 6. Limitations

Synthetic, unreviewed evaluation items and labels; an AI-agent harm review; NLI proxies instead of clinical judgement; seen Track C items; one disclosed Track C item excluded; small safety subsets with wide intervals; base-model truncation at 256 tokens; no measured API latency; no regression check on the remediated pipeline; three open high-severity findings; disk use over budget (E6 §11, D-066).

## 7. Reproducibility

Code, configuration, manifests, aggregate metrics and decision records are in the repository. A fresh clone with a new environment built from `requirements.lock` passed `make lint` and 942 offline tests (D-066). Data builds, indexes, training and evaluations need the local CSV, the GB10 GPU and git-ignored artifacts; the commands are in `docs/data/reproduction.md`, `docs/retrieval/README.md`, `docs/models/` and `docs/evaluation/e4_runbook.md`.
