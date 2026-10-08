# Service metrics and logs

Owner: service-platform-engineer. Code: `src/medquad_qa/observability/`. Scrape `GET /metrics` (Prometheus text format). Prometheus runs in Compose under `--profile monitoring` at 127.0.0.1:9090. Its config is in `configs/monitoring/prometheus.yml` and its alert rules in `configs/monitoring/alerts.yml`.

**No metric measures live answer accuracy.** There are no labels at serving time. Quality numbers come only from labelled offline evaluation runs (`medquad_offline_eval_metric`), each identified by its run ID. Retrieval and answerability scores are uncalibrated and are not medical or diagnostic confidence.

## Catalogue

| Metric | Type | Labels | Source |
|---|---|---|---|
| `medquad_http_requests_total` | counter | method, route (template or `unmatched`), status | middleware |
| `medquad_http_request_duration_seconds` | histogram | method, route | middleware |
| `medquad_http_requests_inflight` | gauge | – | middleware |
| `medquad_qa_requests_total` | counter | mode, outcome (`answered` / `abstained` / `error`) | QAResponse / error mapping |
| `medquad_qa_errors_total` | counter | mode, error_code | error mapping |
| `medquad_qa_latency_seconds` | histogram | mode | `QAResponse.latency_ms` |
| `medquad_qa_component_latency_seconds` | histogram | mode, component (the six contract keys, else `other`) | `component_latency_ms` |
| `medquad_qa_inflight` | gauge | – | generation slots in use |
| `medquad_qa_abstentions_total` | counter | mode, reason (`AbstentionReason`) | QAResponse |
| `medquad_qa_citation_failures_total` | counter | mode, kind (`invalid_ids` / `invalid_citations` / `missing_citations`) | QAResponse |
| `medquad_qa_invalid_citation_ids_total` | counter | mode | count of stripped citation tokens |
| `medquad_qa_citations_per_response` | histogram | mode | answered responses only |
| `medquad_qa_warnings_total` | counter | mode, warning (`lexical_fallback` / `evidence_truncated` / `answerability_fallback` / `evidence_filtered` / `safety_check_failed` / `other`) | QAResponse |
| `medquad_qa_input_chars` | histogram | mode | question length after stripping |
| `medquad_generation_prompt_tokens`, `medquad_generation_completion_tokens` | histogram | mode | `GenerationMeta` |
| `medquad_generation_finish_total` | counter | mode, reason (`stop` / `length`) | `GenerationMeta` |
| `medquad_retrieval_top_score`, `medquad_retrieval_score` | histogram | retriever (`family[:text_mode]`) | `PipelineObserver.on_retrieval` |
| `medquad_retrieval_hits` | histogram | mode | `on_retrieval` |
| `medquad_answerability_score` | histogram | gate | `on_gate` |
| `medquad_answerability_gate_decisions_total` | counter | gate, label (`true` / `false` / `none`) | `on_gate` |
| `medquad_observer_errors_total` | counter | hook | swallowed observer exceptions |
| `medquad_build_info` | info | contracts_version, service_version | startup |
| `medquad_pipeline_state` | gauge (one-hot) | state (`building` / `ready` / `failed`) | pipeline build |
| `medquad_pipeline_build_seconds` | gauge | – | pipeline build |
| `medquad_artifact_info` | gauge (=1) | artifact, version | `pipeline.versions()` |
| `medquad_component_ready` | gauge (0/1) | component, required | `pipeline.readiness()` |
| `medquad_mode_available` | gauge (0/1) | mode | `pipeline.available_modes()` |
| `medquad_offline_eval_metric` | gauge | eval_run_id, track, mode, metric | offline evaluation summary file |
| `medquad_offline_eval_timestamp_seconds` | gauge | eval_run_id | offline evaluation summary file |

### Label cardinality

Label values come from contract literals or allowlists. Unknown values become `other`, and unmatched routes become `unmatched`. Record IDs, questions and free text are never used as label values.

### Counting rule

Abstentions and citation failures are counted once, from the `QAResponse`. The observer hooks `on_abstention` and `on_citations` are deliberately no-ops, to avoid double counting.

## Offline evaluation summary (periodic quality): offline evaluation snapshot, not live accuracy

The evaluator (evaluation-safety-engineer) writes a summary file. The API reads it from `MEDQUAD_OFFLINE_EVAL_PATH`; the Compose default is `artifacts/evaluation/summary/latest_offline_eval.json`. The file is re-read whenever its mtime changes, so no restart is needed.

```json
{"eval_run_id": "e4-20261007-ab12", "completed_at": "2026-10-07T12:00:00+00:00",
 "metrics": [{"track": "B", "mode": "rag", "metric": "citation_validity", "value": 0.97}]}
```

The evaluator writes the file atomically (temporary file + rename). Label values: `track` ∈ {A, B, C, classifier}; `mode` is an experiment mode or a retriever name with `_` in place of `:` (e.g. `hybrid_rrf_answer`); `metric` is snake_case (e.g. `recall_at_5`, `citation_validity`, `p95_latency_ms`). Undefined metrics are omitted, never written as NaN.

Constraints:
- At most 500 rows.
- Values must be finite.
- `eval_run_id`, `track`, `mode` and `metric` must match `[A-Za-z0-9_.:+-]{1,64}`.

An invalid or missing file clears the gauges and never fails a scrape.

## Structured logs

Logs are JSON lines on stdout: `ts`, `level`, `logger`, `msg`, plus fields.

- **Access log** (`medquad_qa.api.access`, `msg=http_request`): request_id, method, route template, status, duration_ms. Query strings and bodies are never logged.
- **QA log** (`medquad_qa.api`, `msg=qa_request`):
  - Always logged: request_id, mode, outcome, abstention_reason or error_code, top_k, question_chars, and counts of retrieved records, citations and invalid citations.
  - Also logged: warnings, pipeline and wall latency, and the model, index and prompt versions.
- **Exceptions** log the exception type only (`exc_type` / `error_type`), because messages and tracebacks could embed request content.
- **Default: no question, answer or evidence text is logged.** `MEDQUAD_LOG_CONTENT=true` adds `question_sha256`, `question_preview`, `answer_sha256` and `answer_preview`, where each preview is the first 200 characters. It also emits a `content_logging_enabled` WARNING at startup. See `docs/operations/runbook.md` for when to use it and how to clean up afterwards.
