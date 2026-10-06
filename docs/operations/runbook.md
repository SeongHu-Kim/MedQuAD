# Operations runbook: local stack

Owner: service-platform-engineer. Scope: one local machine (GB10, aarch64). Every port binds to 127.0.0.1. Nothing is deployed publicly and nothing is pushed.

## 1. Components

| Service | Image | Host port | State | Profile |
|---|---|---|---|---|
| `qdrant` | `qdrant/qdrant:v1.19.1-unprivileged` (digest-pinned) | 127.0.0.1:6333 | volume `medquad_qdrant_storage` | always |
| `mlflow` | `ghcr.io/mlflow/mlflow:v3.16.1` (digest-pinned) | 127.0.0.1:5000 | `mlruns/` (sqlite `mlflow.db` + `artifacts/`) | always |
| `api-gpu` | `medquad-api:gpu`, built from `deploy/docker/api.Dockerfile` with `VARIANT=gpu` | 127.0.0.1:8000 | stateless | `gpu` |
| `api-cpu` | `medquad-api:cpu` (`VARIANT=cpu`) | 127.0.0.1:8000 | stateless | `cpu` |
| `ui` | `medquad-ui:local` (Streamlit, HTTP-only client) | 127.0.0.1:8501 | stateless | always |
| `prometheus` | `prom/prometheus:v3.5.0` (digest-pinned) | 127.0.0.1:9090 | volume `medquad_prometheus_data` | `monitoring` |

Hardening applies to every service:
- read-only root filesystem, with a tmpfs at `/tmp`
- `cap_drop: ALL`
- `no-new-privileges`
- non-root user: API and UI run as uid 10001; MLflow runs as the host uid because it writes `mlruns/`; Qdrant and Prometheus use their images' unprivileged users.

The API mounts `data/`, `artifacts/` and the HF cache read-only, with `HF_HUB_OFFLINE=1`, so no models are downloaded at run time. Images contain code and pinned dependencies only; no data, weights or secrets are baked in.

## 2. Launch

```bash
make gpu-check                 # once per machine/driver change: CDI GPU-in-container check (D-017)
make up PROFILE=gpu            # == docker compose -f deploy/compose.yaml --profile gpu up -d --build
make stack-smoke               # 6-step acceptance sequence; evidence in artifacts/service/stack_smoke/<ts>/
make down                      # stop everything (volumes and mlruns/ are kept)
```

- CPU-only smoke: `make up PROFILE=cpu` (add `MEDQUAD_RETRIEVER=bm25` for a lexical-only, torch-free retriever). The 4B generator on CPU will usually hit the 60 s generation budget (504). It is meant for wiring checks, not for answers.
- Monitoring: add `--profile monitoring`, then open http://127.0.0.1:9090.
- **Option B (no GPU container):** `docker compose -f deploy/compose.yaml up -d qdrant mlflow ui`, then `make api-host`. The API runs as a host process on 127.0.0.1:8000, and the UI container reaches it only if you run the UI on the host instead (`scripts/service/run_ui_host.sh`).

One-time index setup for the Qdrant server (retrieval-engineer's CLI): with `qdrant` up, run `MEDQUAD_QDRANT_URL=http://127.0.0.1:6333 python -m medquad_qa.retrieval build --kind dense`. The collection name equals the dense `index_version`. In Compose, the API uses server mode only (`MEDQUAD_QDRANT_URL=http://qdrant:6333`, `MEDQUAD_QDRANT_PATH` unset), because embedded mode needs a writable lock file.

`TORCH_DISABLE_NATIVE_JIT=1` is set in the API images and in Compose (D-030). torch 2.14 otherwise JIT-compiles some CUDA ops, which needs gcc and Python.h; neither exists in the image or on the host. `scripts/ops/image_cuda_smoke.sh` proves the image path.

Retrieval defaults (D-037, frozen on DEV): both API profiles default to `MEDQUAD_RETRIEVER=dense_fallback` and `MEDQUAD_INDEX_TEXT_MODE=question_answer`. That means dense ranking on `dense-qa-*`, with automatic fallback to `bm25-qa-*` and a `lexical_fallback` warning. Override either variable in the environment for experiments; never change them for frozen TEST runs.

TensorFlow is never installed in the API images (D-036: importing it before Triton segfaults). `scripts/ops/image_smoke.sh` fails if `import tensorflow` succeeds.

Readiness:
- `GET /health/ready` returns 200 once the pipeline is built and every required component (corpus, retriever, base generator) is ok.
- Optional components (finetuned adapter, answerability predictor) show as `degraded: true` and their modes are reported unavailable. Requests for an unavailable mode get 503 `mode_unavailable`, never a silent fallback.

## 3. Health and diagnosis

| Symptom | Check | Typical cause / fix |
|---|---|---|
| `/health/ready` 503 `"pipeline":"failed"` | `failure` field; `docker compose logs api-gpu` (`pipeline_build_failed`, `error_type`) | Missing artifact: corpus (`python -m medquad_qa.data build`), index (retrieval CLI `build`), model weights not in the HF cache. Fix the artifact, then `docker compose restart api-gpu`. |
| `/health/ready` 503 `"status":"starting"` | `medquad_pipeline_state{state="building"}` | Model still loading. Wait. The smoke script allows up to 900 s. |
| 503 `busy` | `medquad_qa_inflight` | One generation slot (`MEDQUAD_MAX_CONCURRENT_GENERATIONS=1`); a previous request (possibly timed out) is still running. Retry after `Retry-After`. |
| 504 `generation_timeout` | `medquad_qa_errors_total{error_code="generation_timeout"}` | Generator exceeded 60 s. CPU profile, or a very large `max_new_tokens`. Use the GPU profile. |
| 503 `artifact_unavailable` mid-traffic | qdrant logs, `curl 127.0.0.1:6333/readyz` | Qdrant down. Hybrid degrades to `lexical_fallback` if the pipeline supports it; otherwise restart qdrant. |
| `warnings: ["lexical_fallback"]` | `medquad_qa_warnings_total` | Dense retriever unavailable; answers are BM25-only. |
| GPU not visible in the container | `make gpu-check` | CDI spec `/var/run/cdi/nvidia.yaml` is regenerated by the NVIDIA toolkit at boot. Do **not** change the Docker runtime config or the socket. If it fails, use option B. |
| HF "Permission denied … trees/…json" warning | api logs | Harmless. The HF cache is read-only by design; weights still load offline. |
| Qdrant ".qdrant-initialized … Read-only file system" warning | qdrant logs | Harmless with a read-only root filesystem; storage is on the volume. |

Logs: `docker compose -f deploy/compose.yaml logs -f api-gpu` prints JSON lines. Filter with `jq 'select(.msg=="qa_request")'`. Correlate using `request_id`, which is also returned in the `X-Request-ID` header.

## 4. Rollback

**Image rollback (API code or dependencies):**
```bash
VARIANT=gpu scripts/ops/rollback.sh snapshot     # BEFORE rebuilding: tag current image as medquad-api:gpu-prev
make up PROFILE=gpu                              # deploy the new build
VARIANT=gpu scripts/ops/rollback.sh rollback     # bad deploy? recreate api-gpu from :gpu-prev (no build), wait for ready
VARIANT=gpu scripts/ops/rollback.sh restore      # roll forward again to :gpu
VARIANT=gpu scripts/ops/rollback.sh status       # image IDs and what is running
```

**Artifact rollback (model, index or corpus):**
- Artifacts are versioned by their manifests (`corpus_version`, `index_version`, `model_version`, `prompt_version`), and those versions are exposed in `/v1/info`, `medquad_artifact_info` and every `QAResponse`.
- To roll back, restore the previous artifact version under `artifacts/` and `data/` with the owner's CLI (e.g. the retrieval `rebuild` command for a given corpus_version), then `docker compose restart api-gpu`.
- Confirm the version labels changed in `/v1/info`.

**MLflow:** `mlruns/mlflow.db` is a single sqlite file. Back it up with `cp mlruns/mlflow.db mlruns/mlflow.db.bak-$(date +%F)` while `mlflow` is stopped.

## 5. Content logging (opt-in, minimised)

- Default: **no question, answer or evidence text in logs.** Only IDs, counts, versions and latencies are logged.
- For a bounded debugging session, set `MEDQUAD_LOG_CONTENT=true` in the API environment (e.g. `MEDQUAD_LOG_CONTENT=true make up`). The API then logs, per request:
  - a sha256 and a 200-character preview of the question and of the answer
  - a `content_logging_enabled` WARNING at startup
- Turn it off afterwards (`make up` without the variable).
- The logs live only in Docker's local json-file logs. Remove them with `docker compose down` followed by `docker compose up -d`, which recreates the containers.
- `scripts/ops/stack_up_smoke.sh` fails its privacy check if question text appears in the logs. Expect that failure while content logging is on.

## 6. Resources and cleanup

- Image sizes (`docker image ls`, 2026-10-06): `medquad-api:gpu` 6.5 GB, `medquad-api:gpu-prev` 6.5 GB (rollback target), `medquad-api:cpu` 1.8 GB, `medquad-ui:local` 0.6 GB. Qdrant, MLflow, Prometheus and the CUDA check image total about 1.9 GB.
- Each GPU rebuild needs about 8 minutes, re-downloads about 5 GB of CUDA wheels, and leaves up to about 15 GB of build cache. After a release, prune with `docker builder prune --all -f`; all build cache on this machine belongs to this project, which was verified on 2026-10-06.
- `medquad_qdrant_storage` holds the server-side dense collections (`dense-answer-*`, `dense-qa-*`), which take about 2 minutes to rebuild. `make down` keeps it.
- Check usage with `docker system df`.
- Never run `docker system prune -a` on a shared machine without asking.
- Volumes: `docker volume ls | grep medquad`. Removing `medquad_qdrant_storage` deletes the server-side vector index; rebuild it with the retrieval CLI.

## 7. Security notes

- No secrets are needed by default. `HF_TOKEN` is only for gated models, which are not used. Never put it in the image; pass it via the environment if it is ever needed.
- Do not change port bindings to `0.0.0.0:`. The containers listen on 0.0.0.0 *inside* the Compose network only.
- The UI renders model and evidence text escaped, with no Markdown links or images and no HTML. Streamlit usage statistics are disabled.
- Threat review and security tests: evaluation-safety-engineer (`docs/security/`, `tests/security/`).
