# S6: real-artifact integration in Docker (GPU profile)

Recorded 2026-10-06 by service-platform-engineer.

**Scope:** this is a technical integration check, not a demo or evaluation. The outputs are not validated. Finding F-002 (safety rules, high) is still open.

**What this file contains:** IDs, versions, counts and timings only. It has no answer or evidence text, by design (D-015). Raw evidence lives under `stack_smoke/20261006T074918Z/` and `logs/`, which are gitignored.

## GPU image (D-027, D-030)

- **Build command:** `MEDQUAD_BUILD_ID=s6-b docker compose -f deploy/compose.yaml --profile gpu build api-gpu`, which exited 0.
- **Build times:**

  | Build | Duration | Cache |
  |---|---|---|
  | First build | 459 s | cold |
  | `s6-a` | 484 s | invalidated: `requirements.lock` changed at 16:38 |
  | `s6-b` | 459 s | invalidated |

- **Images:** `medquad-api:gpu` is 6.5 GB (`docker image ls`), sha256:81ee828c… with label `s6-b`. The rollback target `medquad-api:gpu-prev` is sha256:ce7fc4d3… with label `s6-a`.
- **pip check:** the only line was the known cuSPARSELt platform-tag message, which is benign (D-016).
- **In-image CUDA smoke:** `bash scripts/ops/image_cuda_smoke.sh medquad-api:gpu` exited 0. It used the image's own torch, uid 10001, a read-only rootfs, cap-drop ALL, no-new-privileges, no network and CDI. Output:
  - `2.14.1+cu130 13.0 NVIDIA GB10 TORCH_DISABLE_NATIVE_JIT=1 matmul_ok outer_ok (1, 64, 32) uid=10001`
  - Result JSON: `gpu_image_smoke.json`.
- **Docker disk usage after cleanup** (`docker system df`): images 16.8 GB in total; build cache 14.8 GB, all in use by current images (0 B reclaimable after `docker builder prune -f`).
  - The superseded `medquad-api:gpu` and `medquad-api:cpu` builds and an old UI image were removed by image ID.

## Six-step sequence

**Command:** `PROFILE=gpu MONITORING=1 NO_BUILD=1 READY_TIMEOUT=600 bash scripts/ops/stack_up_smoke.sh`. It exited 0, with overall PASS.

1. **compose config** (`--profile gpu --profile monitoring config -q`): valid.
2. **up -d --no-build:** ok. All five services (qdrant, mlflow, api-gpu, ui, prometheus) were Up. Every published port bound only to 127.0.0.1 (8000, 8501, 5000, 6333, 9090).
3. **Readiness:** live 200; ready 200 after 21 s, with `degraded=true`.
   - Required components were ok: `corpus` medquad-1.0.0-86e384302357, `retriever:bm25` bm25-answer-3c6bf6f998cf, `generator:base` Qwen/Qwen3-4B-Instruct-2507@cdbee75f.
   - `retriever:dense` was ok but optional (hybrid mode).
   - `answerability` was ok, version answerability-lexlr@3defcd00a31f (max-F1 on validation, threshold 0.307).
   - `generator:finetuned` was not ok because no adapter has been published yet, so modes were [base, rag].
4. **Real `POST /v1/qa`:** every response was 200 and generated on the GPU. All four used prompt `rag-v1+df593554` (closed-book prompt `cb-v1+a1f08aaf` for base) and index `bm25-answer-3c6bf6f998cf+dense-answer-4729100e34a8`.

   | request_id | mode | result | citations ⊆ retrieved | invalid IDs | prompt / completion tokens | latency |
   |---|---|---|---|---|---|---|
   | stack-smoke-20261006T074918Z | rag (hybrid_rrf:answer) | abstained `insufficient_evidence` (model sentinel; gate=predictor, score 0.795) | 0 ⊆ 5 | 0 | 1446 / 125 | 23.7 s |
   | s6-extra-1 | rag | answered | 5 ⊆ 5 | 0 | 1342 / 167 | 24.1 s |
   | s6-extra-2 | rag | answered | 4 ⊆ 5 | 0 | 1069 / 166 | 18.8 s |
   | s6-extra-3 | base (closed-book) | answered, no retrieval and no citations | 0 / 0 | 0 | 59 / 120 | 17.0 s |

5. **`/metrics` and `docker compose logs`:**
   - Metrics showed `medquad_qa_requests_total` with rag abstained 1, rag answered 2 and base answered 1, plus `medquad_qa_abstentions_total{reason="insufficient_evidence"}` 1.
   - The `medquad_artifact_info`, `medquad_component_ready` and `medquad_mode_available` series matched readiness.
   - Prometheus target `up{job="medquad-api"}` was 1, and its `sum(medquad_qa_requests_total)` query reflected scraped values.
   - Privacy: the question and answer strings occurred 0 times in the `api-gpu` logs.
   - The UI container reached the API: `http://api:8000/health/ready` returned 200.
   - Memory from `docker stats` after the requests: api-gpu 3.8 GiB RSS (weights live in CUDA unified memory), mlflow 1.8 GiB, qdrant 255 MiB, ui 45 MiB, prometheus 23 MiB.
6. **Summary:** `stack_smoke/20261006T074918Z/summary.md`.

## Rollback drill

Log: `logs/s6_rollback_drill.log`.

1. `VARIANT=gpu scripts/ops/rollback.sh rollback` took 32 s in total, with ready after 25 s. The running image changed from medquad-api:gpu (81ee828c) to medquad-api:gpu-prev (ce7fc4d3). A follow-up rag `POST /v1/qa` returned 200, answered, with 5 citations.
2. `VARIANT=gpu scripts/ops/rollback.sh restore` took 30 s in total, with ready after 25 s. The running image went back to medquad-api:gpu (81ee828c). A follow-up rag `POST /v1/qa` returned 200, answered, with 5 citations.

Note: `s6-a` and `s6-b` were built from the same source. Because the lockfile changed between them, their dependency layers differ too. The drill shows that container recreation from a pinned previous image works, with no rebuild and readiness restored. It does not exercise a behavioural regression.
