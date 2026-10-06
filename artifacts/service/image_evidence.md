# Image and platform evidence (service-platform-engineer)

Recorded 2026-10-06 on GB10 (aarch64), Docker Engine 29.6.2, Compose v5.2.0. All digests are manifest-list digests, verified with `docker buildx imagetools inspect`. Architectures were checked with `docker manifest inspect`.

| Image | Digest | Platforms (manifest) |
|---|---|---|
| python:3.12-slim-bookworm | sha256:7753c33391fc9f01d1984375bf375eb6686d52ba10db6043a86634a5ccf90dcf | 386, amd64, arm/v7, arm64/v8, ppc64le |
| qdrant/qdrant:v1.19.1-unprivileged | sha256:801777072776dc81b2a9dd2007b2ed487571f21ecd30efffd15ddb1671f2193d | amd64, arm64 |
| ghcr.io/mlflow/mlflow:v3.16.1 | sha256:06058cc872276873e9759c635e773342aca4731e240f6d20d88ea43774083984 | amd64, arm64 |
| prom/prometheus:v3.5.0 | sha256:63805ebb8d2b3920190daf1cb14a60871b16fd38bed42b857a3182bc621f4996 | amd64, arm/v7, arm64, ppc64le, s390x |
| nvidia/cuda:13.0.1-base-ubuntu24.04 (GPU check only) | sha256:f8ef28f579ea42a44b415d2c5d46f788e6a9b395c6c83f2929416e1fc192c143 | amd64, arm64 |

## GPU in container (D-017)

`bash scripts/ops/validate_gpu_container.sh` exited 0. The result is in `gpu_validation.json` (gitignored), checked_at 2026-10-06T07:12:28Z.
- **Stage 1:** `nvidia-smi` via CDI `--device nvidia.com/gpu=all`, run as uid 10001 with a read-only rootfs, cap-drop ALL and no network. Output: `NVIDIA GB10, 580.178.04`.
- **Stage 2:** a bf16 matmul with host torch mounted read-only. Output: `2.14.1+cu130 13.0 NVIDIA GB10 matmul_ok`.
- **Compose:** the `devices: [nvidia.com/gpu=all]` syntax was verified with a throwaway compose project running `nvidia-smi`, which printed `NVIDIA GB10, 580.178.04`.

## Local builds

- `docker compose -f deploy/compose.yaml --profile cpu build api-cpu ui` exited 0 in 1m47s. Results: `medquad-api:cpu` 1.8 GB and `medquad-ui:local` 607 MB; `pip check` in both reported "No broken requirements found". Log: `logs/build_cpu_ui.log`.
- `bash scripts/ops/image_smoke.sh medquad-api:cpu` exited 0:
  - live 200
  - ready 503 (no artifacts mounted)
  - POST 503 not_ready
  - invalid POST 422
  - /metrics exposes medquad_pipeline_state
  - the container runs as uid 10001
