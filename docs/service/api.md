# MedQuAD QA API

Owner: service-platform-engineer. Code: `src/medquad_qa/api/`. Factory: `medquad_qa.api.app.create_app(settings=None, pipeline_provider=None)`.
Research prototype for general medical information only. It is not medical advice and has not been clinically validated.

## Running

| How | Command | Binds |
|---|---|---|
| Host process (dev, or D-017 option B) | `scripts/service/run_api_host.sh` (or `make api-host`) | 127.0.0.1:8000 |
| Compose, GPU | `docker compose -f deploy/compose.yaml --profile gpu up -d --build` | 127.0.0.1:8000 |
| Compose, CPU smoke | `docker compose -f deploy/compose.yaml --profile cpu up -d --build` | 127.0.0.1:8000 |

The interactive OpenAPI page is at `http://127.0.0.1:8000/docs` once the service is running.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/qa` | Answer a question in an explicit experiment mode (`QARequest` → `QAResponse`, contracts v1.1.x). |
| GET | `/v1/info` | Service and contract versions, available modes, loaded artifact versions, input limits, disclaimer. |
| GET | `/health/live` | Process is up. Always 200 while the event loop runs. |
| GET | `/health/ready` | 200 only when the pipeline is built, every `required` component is ok, and at least one mode is servable. Otherwise 503 with `Retry-After`. |
| GET | `/metrics` | Prometheus text format (see `metrics.md`). |

### Request (`QARequest`)

```json
{"question": "What are the symptoms of glaucoma?", "experiment_mode": "rag", "top_k": 5,
 "generation": {"max_new_tokens": 256, "temperature": 0.0}}
```

- `experiment_mode` is one of `base`, `rag`, `finetuned` or `finetuned_rag`. The default is `rag`. Modes are never switched implicitly; an unavailable mode returns 503 `mode_unavailable`.
- Bounds come from the shared contracts:
  - `question` is 3–2000 characters after stripping, and control characters are rejected.
  - `top_k` is 1–20.
  - `max_new_tokens` is 1–1024.
  - `temperature` is 0–1.5.
  - `top_p` is in (0, 1].
  - Unknown fields are rejected.
- The request body is capped at 16 KB (`MEDQUAD_MAX_BODY_BYTES`). Larger bodies are rejected with 413 before parsing.

### Response

The response is `QAResponse`, defined in `src/medquad_qa/contracts/qa.py`. It contains:
- the answer, or an abstention with its reason
- citations, which are always a subset of `retrieved_record_ids`
- `invalid_citation_ids`, listing citation tokens that were stripped
- warnings, such as `lexical_fallback` or `evidence_truncated`
- model, corpus, index and prompt versions
- end-to-end and per-component latency
- generation token accounting
- the disclaimer

`X-Request-ID` is echoed on every response. A client-supplied value is kept only if it matches `^[A-Za-z0-9._-]{1,64}$`; otherwise a new 32-hex ID is generated.

### Errors

Every error body has the same shape: `{"error_code", "message", "request_id"}`, plus an `errors` list for 422s. Bodies never echo request content or internal exception messages.

| Status | `error_code` | Cause |
|---|---|---|
| 413 | `payload_too_large` | Body larger than `MEDQUAD_MAX_BODY_BYTES` |
| 422 | `validation_error` | Contract validation failed. `errors` gives `loc` and `type` only, never the input. |
| 503 + Retry-After | `not_ready` | Pipeline still building, or the build failed (see `/health/ready`) |
| 503 + Retry-After | `mode_unavailable` | Requested mode not servable (e.g. LoRA adapter missing) |
| 503 + Retry-After | `artifact_unavailable` | `ArtifactUnavailableError` / `RetrieverUnavailableError` (e.g. Qdrant down) |
| 503 + Retry-After | `busy` | No generation slot was free within `MEDQUAD_QUEUE_TIMEOUT_S` (10 s) |
| 504 | `generation_timeout` | Generator-internal budget exceeded (60 s, model-engineer) |
| 504 | `request_timeout` | API backstop `MEDQUAD_REQUEST_TIMEOUT_S` (120 s) exceeded |
| 502 | `generation_failed` | `GenerationError` (e.g. prompt over `max_input_tokens`) |
| 500 | `internal_error` | `ContractViolationError` or an unexpected exception. Only the exception type is logged. |

## Concurrency and timeouts

- One global generation semaphore (`MEDQUAD_MAX_CONCURRENT_GENERATIONS`, default 1) covers all modes, because every mode generates on the same GPU.
- `pipeline.answer` runs in a worker thread. After a 504, the slot stays held until that thread actually returns, because the GPU is still busy. Requests arriving in the meantime receive 503 `busy`.
- The pipeline is built in a background thread at startup. Liveness is immediate, and readiness follows the build. A failed build keeps the process alive, and `/health/ready` reports `"pipeline": "failed"` with the exception type and a short message.

Sizing (model-engineer bench, `artifacts/models/bench/generator_base_cuda.json`, GB10 bf16):
- Generator load takes about 6 s.
- Peak CUDA memory is about 8.5 GiB, and peak process RSS during load is about 12 GiB, so budget at least 16 GiB for the API container.
- Throughput is about 20 tokens/s; 256 new tokens take a median 12.9 s.
- The generator's 60 s deadline includes queue time, so `max_new_tokens` above about 1000 will usually return 504 `generation_timeout`.

## Configuration (`MEDQUAD_*` environment)

| Variable | Default | Meaning |
|---|---|---|
| `MEDQUAD_BIND_HOST` / `MEDQUAD_API_PORT` | `127.0.0.1` / `8000` | uvicorn bind. In containers this is `0.0.0.0`; the host publish stays on 127.0.0.1. |
| `MEDQUAD_MAX_BODY_BYTES` | 16384 | Body limit (413) |
| `MEDQUAD_REQUEST_TIMEOUT_S` | 120 | API backstop (504) |
| `MEDQUAD_QUEUE_TIMEOUT_S` | 10 | Maximum wait for a generation slot (503 busy) |
| `MEDQUAD_MAX_CONCURRENT_GENERATIONS` | 1 | Global semaphore size |
| `MEDQUAD_RETRY_AFTER_S` | 5 | `Retry-After` on 503 responses |
| `MEDQUAD_LOG_LEVEL` | INFO | Log level |
| `MEDQUAD_LOG_CONTENT` | false | Opt-in content logging (see runbook) |
| `MEDQUAD_OFFLINE_EVAL_PATH` | unset | Offline evaluation summary JSON exported as gauges |

Pipeline settings (retriever, model, index paths) are read by the owners' factories. See `docs/retrieval/` and `docs/models/`.

## Tests

`pytest tests/api tests/observability` runs offline against a synthetic `FakePipeline` (`tests/api/api_fakes.py`). No models or data are used.
