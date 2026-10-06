# Shared Contracts (v1.1.1)

Owner: lead. Source of truth: `src/medquad_qa/contracts/`. Change log: `docs/decisions.md` (D-003, D-009, D-021–D-023).
Owners propose changes by message; they never edit contracts directly.

| Module | Contents |
|---|---|
| `records.py` | `Provenance`, `MedicalRecord` (raw + normalized text, `record_id = mq-<16 hex>`, group IDs, `quality_flags`) |
| `qa.py` | `RetrievalHit`, `GenerationParams`, `QARequest` (bounded, sanitised), `Citation`, `AbstentionReason`, `GenerationMeta`, `QAResponse`, limits, `REQUEST_ID_PATTERN` |
| `answerability.py` | `PredictorOutput` (non-clinical; max aggregation), `AnswerabilityPair` |
| `evaluation.py` | `EvaluationExample` (tracks, label/question provenance, `eval_split` dev/test/train_probe, reviewer) |
| `interfaces.py` | Errors, `ChatMessage`, `GenerationResult`, `ComponentStatus`, and the Protocols `Retriever`, `Generator`, `BatchGenerator`, `AnswerabilityPredictor`, `ReadinessReporter`, `QAPipeline`, `PipelineObserver` |

## Error → HTTP mapping (API)

| Error | Status |
|---|---|
| `ModeUnavailableError` | 503, `error_code=mode_unavailable` |
| `ArtifactUnavailableError` / `RetrieverUnavailableError` | 503, Retry-After |
| `GenerationTimeoutError` (catch before its parent `GenerationError`) | 504 |
| `GenerationError` | 502 |
| `ContractViolationError` / unexpected | 500 (generic message) |
| Pydantic validation | 422 (never echoes input) |

## Factories (implemented by owners)

- `medquad_qa.models.load_generator(variant: Literal["base","finetuned"], settings=None) -> Generator`, from model-engineer.
- `medquad_qa.rag.factory.build_pipeline(settings=None, *, observer=None) -> QAPipeline & ReadinessReporter`, from retrieval-engineer.
- `medquad_qa.api.app.create_app(settings=None, pipeline_provider=None) -> FastAPI`, from service-platform-engineer.

## Key semantics

- **Citations (D-010):** the model emits `[E#]`. The pipeline rewrites these to `[mq-…]` and enforces citations ⊆ supplied ⊆ retrieved.
- **Safety rules (D-023):** personalized-advice rules run in all modes.
- **Evidence budget (D-022):** evidence is fitted to the 3072-token input budget by dropping whole low-ranked hits, with the warning `evidence_truncated`.
- **Scores:** retrieval scores are uncalibrated and never presented as medical confidence. Answerability is not diagnostic confidence.
