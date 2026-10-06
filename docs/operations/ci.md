# CI

Owner: service-platform-engineer. Workflow: `.github/workflows/ci.yml`.

**Status:** this repository has no Git remote, so the workflow has **never run on GitHub Actions**. The local `make ci` run (and the scripts below) is the evidence of record. Do not cite a GitHub run.

| Job | What it runs | Runners |
|---|---|---|
| `test` | CPU torch from the PyTorch CPU index, plus the lock minus the CUDA stack, then `ruff check`, `ruff format --check`, `mypy src/medquad_qa`, and `pytest -m "not gpu and not real_model and not real_data and not docker and not slow"` | ubuntu-24.04 (x86_64), ubuntu-24.04-arm |
| `compose-and-images` | `docker compose config -q` for the cpu, gpu and monitoring profiles; build `medquad-api:ci-cpu` and `medquad-ui:ci` (no push); `scripts/ops/image_smoke.sh` | ubuntu-24.04-arm |
| `pip-audit` | Dependency audit of the lock. Non-blocking (`continue-on-error`). | ubuntu-24.04 |

Constraints:
- No dataset, no model downloads (`HF_HUB_OFFLINE=1`), no GPU, no secrets.
- `permissions: contents: read`.
- Nothing is pushed or published.

Local equivalents:

```bash
make ci                                                   # lint + typecheck + offline tests + compose config
for p in cpu gpu monitoring; do docker compose -f deploy/compose.yaml --profile $p config -q; done
docker build -f deploy/docker/api.Dockerfile --build-arg VARIANT=cpu -t medquad-api:cpu .
bash scripts/ops/image_smoke.sh medquad-api:cpu           # live 200, ready 503 without artifacts, 422, /metrics
```

GPU, real-model and Docker-stack checks are kept out of CI. They run only on the GB10 via `make gpu-check` and `make stack-smoke`.
