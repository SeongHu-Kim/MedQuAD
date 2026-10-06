# Architecture Overview

Owner: lead. Contracts are in `docs/architecture/contracts.md`; decisions are in `docs/decisions.md`.

```
medquad.csv ──► data (ingest → normalize → audit → group/split → exports)
                  │ corpus.jsonl + manifests (corpus_version, split_version)
                  ▼
        retrieval (BM25 · dense bge-small/Qdrant · hybrid RRF)  ──► index manifests (index_version)
                  │ RetrievalHit[]
                  ▼
QARequest ─► rag pipeline (langchain-core LCEL):
               safety rules (all modes) → retrieve → answerability gate → evidence budget
               → prompt [E#] → Generator (Qwen3-4B base | +LoRA) → citation validation → QAResponse
                  ▲                         ▲
          models: AnswerabilityPredictor     models: load_generator(base|finetuned), LoRA training
                  │
api (FastAPI /v1/qa, /health/*, /metrics) ◄── ui (Streamlit over HTTP)
                  │
observability (Prometheus metrics, JSON logs w/o content) · MLflow (runs) · Docker Compose (127.0.0.1)
                  │
evaluation (frozen eval sets, retrieval/QA/classifier metrics, 4-way comparison, security tests)
```

## Adapter boundaries

| Concern | Package | Owner | Contract |
|---|---|---|---|
| Dataset ingestion | `medquad_qa.data` | data-steward | `MedicalRecord` |
| Retrieval | `medquad_qa.retrieval` | retrieval-engineer | `Retriever` → `RetrievalHit` |
| Orchestration | `medquad_qa.rag` | retrieval-engineer | `QAPipeline`, `ReadinessReporter` |
| Generation / training | `medquad_qa.models`, `.training` | model-engineer | `Generator`, `BatchGenerator`, `AnswerabilityPredictor` |
| Transport / UI | `medquad_qa.api`, `.ui` | service-platform-engineer | `QARequest` / `QAResponse` |
| Monitoring | `medquad_qa.observability` | service-platform-engineer | `PipelineObserver` |
| Evaluation | `medquad_qa.evaluation` | evaluation-safety-engineer | `EvaluationExample`, `AnswerabilityPair` |

## Experiment modes

`base` (no retrieval), `rag`, `finetuned` (LoRA, no retrieval), `finetuned_rag`. These modes are experiment configurations. They make no claim of clinical reliability.

## Version identifiers

- `corpus_version`, `split_version`: data manifests
- `index_version`: retrieval manifests
- `model_version`: `<hf_id>@<rev>[+lora:<run_id>]`
- `prompt_version`: `rag-v1+<sha8>`
- `threshold_version`: answerability
- `CONTRACTS_VERSION`: 1.1.1
