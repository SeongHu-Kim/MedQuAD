# E4 serving image: build, smoke and launch record

Owner: service-platform-engineer. Decision record: docs/decisions.md D-049 (consistent with this file).

Scope: technical integration check of the serving image only. It is not an evaluation, and no output here is validated.

This file holds IDs, versions, counts and timings only, with no question, answer or evidence text. Raw evidence stays git-ignored under `artifacts/service/stack_smoke/` and `artifacts/service/logs/`.

All times are UTC.

## 1. Timeline (each item cites its evidence)

1. **10:17:13Z: interrupted earlier build attempt.** Its log ends here and no image was produced. The log is kept at `artifacts/service/logs/e4_image_build.log`.
2. **10:32:29Z–10:40:29Z: build.** Command: `bash scripts/ops/e4_build.sh` (user-reviewed wrapper).
   - Status line: `STATUS build_exit=0 tee_exit=0 tail_exit=0 elapsed_s=480`
   - Log: `artifacts/service/logs/e4_build_619228d_20261006T103229Z.log`
   - Image created 10:40:18Z: `medquad-api:gpu` = `sha256:7aa365e7ff600c87024243149754b637ebdd388b1392f59310e53d468bcdc338`
   - Label `medquad.build_id=619228daaf58`. Source commit: `619228daaf58a52ac38f8452f00ed3e967a4f308`, exported with `git archive`.
3. **10:45:12Z: CURRENT CUDA smoke (image 7aa365e7).** Command: `scripts/ops/image_cuda_smoke.sh medquad-api:gpu`. Result in `artifacts/service/gpu_image_smoke.json`:
   - image_id `sha256:7aa365e7ff600c87024243149754b637ebdd388b1392f59310e53d468bcdc338`
   - exit_code 0, result pass
   - output `2.14.1+cu130 13.0 NVIDIA GB10 TORCH_DISABLE_NATIVE_JIT=1 matmul_ok outer_ok (1, 64, 32) uid=10001`
4. **10:49:04Z: containers created** for api-gpu, ui, qdrant and mlflow (`docker inspect` Created/StartedAt).
   - Reported launch command, approved by the user: `docker compose -f deploy/compose.yaml --profile gpu up -d --no-build --pull never`, reported as returning `UP_EXIT=0`.
   - No launch log was saved, so the flags are not independently evidenced. The image's creation time (10:40:18Z) preceding the containers' (10:49:04Z) is consistent with no build at launch.
5. **10:49:53Z: `ready.json` and `info.json` captured.** `info.json` lists the available modes base, finetuned, finetuned_rag and rag, with these versions:
   - retriever `dense:qa`, index `dense-qa-dc6b6a345fca`
   - prompts `rag-v1+df593554` and `cb-v1+a1f08aaf`
   - safety `safety-v2+43836b6c`
   - gate `answerability-lexlr@3defcd00a31f:maxf1-val:0.307172`
   - base `Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554`
   - finetuned `Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554+lora:sft-main-20261006-081347-3f16e30e`
6. **10:51:12Z–10:51:33Z: one QA request per mode.** Saved as `qa_rag.json`, `qa_finetuned_rag.json`, `qa_base.json` and `qa_finetuned.json`.
7. **10:54:19Z: `metrics.txt` and `compose_logs.txt` captured.**

The files from items 5–7 are in `artifacts/service/stack_smoke/e4_20261006T104953Z/`.

## 2. Running-service identities

Checked with lead `docker inspect` at about 11:00Z.

- api-gpu: `sha256:7aa365e7ff600c87024243149754b637ebdd388b1392f59310e53d468bcdc338` (`medquad-api:gpu`)
- ui: `sha256:fe1d753c4a4845880680321fa0d20f4d321a7d6d0d70c365c36719bd589e8bd7` (`medquad-ui:local`)
- qdrant: `sha256:c2b44393f4e37c1f6bb6e6985803ea238689934dbdd91873a067d22647699046`
  - repo digest `qdrant/qdrant@sha256:801777072776dc81b2a9dd2007b2ed487571f21ecd30efffd15ddb1671f2193d`
- mlflow: `sha256:302f805c7382e0fa546687b283fd448428ee55702faa7e97b9a31848594308ee`
  - repo digest `ghcr.io/mlflow/mlflow@sha256:06058cc872276873e9759c635e773342aca4731e240f6d20d88ea43774083984`

## 3. Per-mode QA results

Taken from the saved response bodies.

- **base:** answered. No citations (closed-book). model_version is the base model; prompt_version `cb-v1+a1f08aaf`.
- **rag:** answered. 2 citations, all within its retrieved record IDs. Base model; prompt_version `rag-v1+df593554`.
- **finetuned:** answered. No citations (closed-book). model_version `…+lora:sft-main-20261006-081347-3f16e30e`; prompt_version `cb-v1+a1f08aaf`.
- **finetuned_rag:** **abstained, abstention_reason `missing_citations`.** model_version `…+lora:sft-main-20261006-081347-3f16e30e`; prompt_version `rag-v1+df593554`. This outcome is pipeline-correct and consistent with closed-book SFT (D-013); E4 measures how often it happens.
- **All four:** 0 invalid citation IDs.
- **`compose_logs.txt`:** 0 occurrences of `lexical_fallback`.

## 4. Not evidenced by this check

- **HTTP status codes:** the saved files contain response bodies only.
- **Per-request GPU execution:** the smoke proves only that CUDA works inside the container.
- **Log privacy:** not assessed for this run.

## 5. Historical S6 evidence (image 81ee828c), kept separate from section 1

The original S6 smoke JSON, `artifacts/service/gpu_image_smoke.json`, recorded image 81ee828c with checked_at 2026-10-06T07:53:30Z and result pass (as reported by service-platform-engineer). That JSON is **lost**:

- It was overwritten at 10:45:12Z by the current smoke in section 1, item 3. No backup was made beforehand.
- It was never committed, because `artifacts/service/*` ignores non-`.md` files.
- No backup was made afterwards, because a copy would now mislabel the 7aa365e7 result.
- The S6 result and output line survive only in the committed `artifacts/service/s6_integration.md` (commit 1ae8e12).

## 6. Images and rollback targets after cleanup

Checked with lead `docker image inspect` at 11:04Z.

- `sha256:7aa365e7ff600c87024243149754b637ebdd388b1392f59310e53d468bcdc338`: `medquad-api:gpu` (E4 serving image)
- `sha256:81ee828c2b171c6858b161aca3fec8a3da634e1593a62e3132e58e003e798f00`: `medquad-api:gpu-prev`, `medquad-api:gpu-s6b-81ee828c`. Label `s6-b`, built 16:48:40 KST. This is the image verified in S6. It is historical evidence only and **pre-fix** (D-050).
- `sha256:ce7fc4d3bfa68388ff9ce8b5f5ab509f834fd999d8f66ef617bbae281c350050`: untagged. Label `s6-a`, built 16:40:47 KST. It was used in the S6 rollback drill. It is historical evidence only and **pre-fix** (D-050).
- `sha256:a4ad3b03b17d6562233594ef1c443038b5e8bdd3842c4325dc0c2a9c3b959d0e`: `medquad-api:cpu`
- `sha256:fe1d753c4a4845880680321fa0d20f4d321a7d6d0d70c365c36719bd589e8bd7`: `medquad-ui:local`
- Pinned service images, unchanged:
  - qdrant `sha256:c2b44393f4e37c1f6bb6e6985803ea238689934dbdd91873a067d22647699046`
  - mlflow `sha256:302f805c7382e0fa546687b283fd448428ee55702faa7e97b9a31848594308ee`
  - python `sha256:eb7d16fec874f15b584063275012ac719885a2b87574d124c0da9c156d504629`
  - cuda `sha256:bd59924159b366ede44d333b4ba86e8f0021529336f488a7b3506f8f919a0a5e`
  - prometheus `sha256:5a1d4a6d0adfd8f785bb0f1243f2a90457e508eaf5eb7e958aa5ea1463020fcb`

Neither S6 image is an approved E4 serving rollback target. Both predate the safety-fix commit `0e4ed7e` (16:57:32 KST), so serving either would reintroduce resolved findings, including F-002 (high). `scripts/ops/rollback.sh rollback` uses `medquad-api:gpu-prev`, which resolves to the pre-fix 81ee828c, so it must not be used for E4 serving rollback (D-050).

## 7. Build-cache cleanup

Command: `docker buildx prune --builder default --all --force`, run once with user approval, 11:02Z–11:04Z.

- Build cache: 15.31 GB → 2.64 GB (12.67 GB reclaimed).
- Filesystem used (`df -h /`): 91 GB → 85 GB.
- Measured project size after cleanup: about 46 GB, over the ~40 GB budget. It breaks down as:
  - repo directory incl. `.venv`: 11 GB
  - `~/.cache/huggingface`: 8.2 GB
  - Docker: 27.1 GB (images 23.15 GB, build cache 2.64 GB, volumes 1.31 GB)
- No images, containers or volumes were removed.

## 8. Shutdown

Run with one-time user approval: `bash e4_shutdown_verify.sh` (lead scratchpad script). It ran `docker compose -f deploy/compose.yaml --profile gpu down`, with no `-v`, `--volumes` or `--rmi`, no prune and no `make`.

- 12:11:37Z–12:11:40Z: `DOWN_EXIT=0`. Containers removed: ui, mlflow, api-gpu, qdrant. Network removed: `medquad_default`.
- Verification (read-only, same script):
  - no containers with label `com.docker.compose.project=medquad` remain;
  - `nvidia-smi --query-compute-apps` reports none (exit 0);
  - all 10 image IDs in section 6 are present;
  - volumes `medquad_qdrant_storage`, `medquad_qdrant_snapshots` and `medquad_prometheus_data` are present.
- Result: `VERIFY_FAIL=0`, "shutdown and verification passed".
- Log (git-ignored): `artifacts/service/logs/e4_shutdown_20261006T121137Z.log`.
