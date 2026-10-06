# MedQuAD Evidence-Grounded Medical QA

A **non-clinical research prototype**. It answers general medical-information questions from the MedQuAD corpus (NIH sources), cites the retrieved records it used, and abstains when evidence is inadequate or when a request asks for personal medical advice. It compares a base LLM, RAG, a LoRA-fine-tuned LLM, and fine-tuned RAG under matched settings.

> **Not medical advice. Not clinically validated.** No clinician reviewed any output. Answerability and retrieval scores are not medical confidence.

**Status (2026-10-06):** in progress. The live ledger is in [docs/project-status.md](docs/project-status.md) and decisions are in [docs/decisions.md](docs/decisions.md). Every result below links to a saved artifact. Items marked _pending_ have not been run yet.

## What it does

| Component | Implementation | Evidence |
|---|---|---|
| Data | MedQuAD Kaggle CSV → 16,336 records + 76 documented exclusions; leakage-resistant group split (13,014 / 1,665 / 1,657), independently verified | [docs/data/](docs/data/), `data/manifests/` |
| Retrieval | BM25 (bm25s), dense (BAAI/bge-small-en-v1.5) in **Qdrant**, hybrid RRF, optional cross-encoder; frozen RAG config `dense_fallback` on the question+answer index | [docs/retrieval/README.md](docs/retrieval/README.md) |
| Generation | Qwen/Qwen3-4B-Instruct-2507 (Apache-2.0, pinned revision), bf16 on NVIDIA GB10 | [docs/models/](docs/models/) |
| Fine-tuning | LoRA (PEFT, r16/α32) SFT on the train split only, closed-book | _pending: M4 run in progress_ |
| Grounding | LangChain-core pipeline: safety rules → retrieve → answerability gate → prompt → generate → citation validation (cited ⊆ supplied ⊆ retrieved) | `src/medquad_qa/rag/` |
| Supervised baseline | Answerability classifier: logistic regression baseline + **TensorFlow/Keras** BiGRU (CPU) | _pending: Keras result_ |
| Service | FastAPI (`/v1/qa`, `/health/live`, `/health/ready`, `/metrics`), Streamlit demo, Prometheus metrics, MLflow, Docker Compose (arm64, 127.0.0.1-only, non-root) | [docs/service/](docs/service/), [docs/operations/](docs/operations/) |
| Evaluation | Frozen eval sets (Track A 300, Track C 240, probe 100, DEV 60); retrieval, citation, abstention, classifier and latency metrics; security tests | [docs/evaluation/](docs/evaluation/), [docs/security/](docs/security/) |

## Quick start

```bash
make venv && make install        # CUDA 13 torch (aarch64) + extras; see requirements.lock
make test                        # offline tests (no GPU, models or Docker)
python -m medquad_qa.data build  # requires local medquad.csv (not redistributed)
make up PROFILE=gpu              # local stack on 127.0.0.1 (needs built images + artifacts)
```

The full reproduction steps are in [docs/data/reproduction.md](docs/data/reproduction.md), [docs/retrieval/README.md](docs/retrieval/README.md), [docs/models/](docs/models/) and [docs/operations/runbook.md](docs/operations/runbook.md).

## Results

_Pending: E4 controlled comparison._ This section will contain only numbers copied from `artifacts/evaluation/` with their commands.

## Limitations (known so far)

- **Dataset licence:** the Kaggle export's slug, version and licence are unverified. Use it locally for research and do not redistribute it.
- **Metadata:** the export has no source URLs or document IDs, so citations are record-level, and grouping is content- and topic-based rather than document-based.
- **Index contents:** the question+answer index contains the original MedQuAD questions, so headline retrieval numbers are labelled accordingly, and answer-only variants are reported alongside them.
- **Safety rules:** the personalized-advice rules are heuristics and will miss some cases. Held-out Track C gives the measured rate.
- **Labels:** agent-written evaluation questions are labelled `llm_generated_unreviewed`. No item is human-reviewed unless the user reviews it.
- **Environment workarounds:** `TORCH_DISABLE_NATIVE_JIT=1` (Python.h is absent; D-030), and TensorFlow runs in a separate process (D-036).
