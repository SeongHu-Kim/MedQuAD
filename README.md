# MedQuAD Evidence-Grounded Medical QA

A **non-clinical research prototype**. It answers general medical-information questions from the MedQuAD corpus (NIH sources), cites the retrieved records it used, and abstains when evidence is inadequate or when a request asks for personal medical advice. It compares a base LLM, RAG, a LoRA-fine-tuned LLM, and fine-tuned RAG under matched, pre-declared evaluations.

> **Not medical advice. Not clinically validated.** No clinician reviewed any output. Answerability and retrieval scores are not medical confidence.

**Status (2026-10-08): complete as a research prototype, with three open high-severity findings.** F-009, F-010 and F-011 are open and accepted by the user as documented known limitations (D-066). **Demo, deployment and user-facing serving are blocked** (D-052). The ledger is [docs/project-status.md](docs/project-status.md), decisions are in [docs/decisions.md](docs/decisions.md), and the independent evaluation is [docs/evaluation/e6_report.md](docs/evaluation/e6_report.md). Portfolio documents are in [docs/portfolio/](docs/portfolio/).

## What it does

| Component | Implementation | Evidence |
|---|---|---|
| Data | MedQuAD Kaggle CSV → 16,336 records + 76 documented exclusions; leakage-resistant group split (13,014 / 1,665 / 1,657), independently verified | [docs/data/](docs/data/), `data/manifests/` |
| Retrieval | BM25 (bm25s), dense (BAAI/bge-small-en-v1.5) in **Qdrant** (embedded), hybrid RRF, optional cross-encoder; frozen config: dense over the question+answer index | [docs/retrieval/README.md](docs/retrieval/README.md) |
| Generation | Qwen/Qwen3-4B-Instruct-2507 (Apache-2.0, pinned revision), bf16 on NVIDIA GB10 | [docs/models/](docs/models/) |
| Fine-tuning | LoRA (PEFT) SFT on the train split: v1 closed-book (in use) and v2c citation-formatted (not promoted) | [docs/models/training.md](docs/models/training.md), `artifacts/models/runs/` |
| Grounding | langchain-core pipeline: safety rules (all modes) → retrieve → evidence filter → answerability gate → prompt → generate → citation validation (cited ⊆ supplied ⊆ retrieved) | `src/medquad_qa/rag/`, [docs/portfolio/architecture.md](docs/portfolio/architecture.md) |
| Supervised baseline | Answerability classifier: logistic regression (served gate) + **TensorFlow/Keras** BiGRU | [docs/models/answerability_card.md](docs/models/answerability_card.md) |
| Service | FastAPI (`/v1/qa`, `/v1/info`, `/health/live`, `/health/ready`, `/metrics`), Streamlit demo, Prometheus metrics, MLflow, Docker Compose (arm64, 127.0.0.1-only, non-root) | [docs/service/](docs/service/), [docs/operations/](docs/operations/) |
| Evaluation | Frozen sets (DEV 60, Track A 300, Track C 240, probe 100), pre-declared analyses, security tests, a hash-registered fresh retest | [docs/evaluation/](docs/evaluation/), [docs/security/](docs/security/) |

## Quick start

```bash
make venv && make install        # CUDA 13 torch (aarch64) + extras; see requirements.lock
make lint && make test           # offline: no GPU, models or Docker needed
python -m medquad_qa.data build  # requires a local medquad.csv (not redistributed)
make up PROFILE=gpu              # local stack on 127.0.0.1; blocked for demos while D-052 stands
```

A fresh clone with a new environment built from `requirements.lock` passed `make lint` and 942 offline tests (D-066). Data builds, indexes, training and evaluations need the local CSV, the GPU and git-ignored artifacts; see [docs/data/reproduction.md](docs/data/reproduction.md), [docs/retrieval/README.md](docs/retrieval/README.md), [docs/models/](docs/models/), [docs/evaluation/e4_runbook.md](docs/evaluation/e4_runbook.md) and [docs/operations/runbook.md](docs/operations/runbook.md).

## Results

All numbers are from [E6](docs/evaluation/e6_report.md); `[a, b]` are 95% intervals. Summary: [docs/portfolio/experiment_comparison.md](docs/portfolio/experiment_comparison.md).

- **Retrieval** (Track A, n = 300): dense question+answer index R@5 0.950, the best of 8 variants. Optimistic, because the index contains the original questions; 0.92 on AI-written questions.
- **Four modes** (Track A): answered 300 / 243 / 300 / 7 for base / rag / finetuned / finetuned_rag. No reference-coverage difference was detected for rag vs base (−0.005 [−0.051, +0.042]) or finetuned vs base (−0.030 [−0.075, +0.015]). finetuned_rag with v1 failed for missing citations.
- **Adapter v2c:** fixed the citation failure (225/300 answered) but copies single records (copy rate 0.9996), so its metric gains are not evidence of better answers.
- **Remediation retest** (fresh held-out probes, run once): evidence injection still leaks, rag 13/40 [0.201, 0.480] and v2c 19/40 [0.329, 0.625] against a bar of ≤ 1/40. Personal-advice and crisis handling scored 60/60, 30/30 and 0/60 over-refusal, but F-009 stays unresolved because the pre-declared owner bar was missed first (D-063, D-065).

## Limitations

- **Open high findings:** F-009 (personal advice/crisis), F-010 (instructions in evidence followed by rag), F-011 (v2c copies planted text). See [docs/portfolio/safety_security_report.md](docs/portfolio/safety_security_report.md).
- **Labels:** every evaluation question and label is template-generated or AI-written (`llm_generated_unreviewed`); none is human-reviewed. The fresh-retest harm review was done by an AI agent.
- **No clinical validation:** metrics measure agreement with MedQuAD reference answers and supplied evidence, not medical correctness.
- **Not measured:** API latency (the S3 smoke was blocked), and the regression check on the remediated pipeline (not run by the pre-agreed rule).
- **Dataset licence:** the Kaggle export's slug, version and licence are unverified. Use it locally for research and do not redistribute it.
- **Metadata:** no source URLs or document IDs, so citations are record-level and grouping is content- and topic-based.
- **Index contents:** the question+answer index contains the original MedQuAD questions; answer-only variants are reported beside it.
- **Environment:** `TORCH_DISABLE_NATIVE_JIT=1` (D-030); TensorFlow runs in a separate process (D-036); disk use exceeds the ~40 GB budget (D-066).
