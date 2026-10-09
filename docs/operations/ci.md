# CI

Owner: service-platform-engineer. Workflow: `.github/workflows/ci.yml`.

**Status (D-081):** runs are on GitHub Actions (private repository). Latest: https://github.com/SeongHu-Kim/MedQuAD/actions/runs/37930186959 on `91fa90f`, all jobs green:
- ubuntu-24.04 (x86, job 113818717232): ruff, mypy and the offline suite passed (973 passed, 3 skipped, 6 deselected). Skipped: the Keras test and the two HashingEmbedder golden tests (aarch64-only, D-080). The exact-embedder goldens and the tie-free check pass, so these goldens are identical on both runners.
- ubuntu-24.04-arm (job 113818717268): 975 passed, 1 skipped (Keras), 6 deselected.
- Compose config and the CPU image build and smoke passed; this was the first image build with setuptools 83.0.0.
- pip-audit passed (only the accepted nltk advisory is ignored).

History:
- Run https://github.com/SeongHu-Kim/MedQuAD/actions/runs/37917163077 (D-079): ubuntu-24.04-arm passed ruff, mypy and the offline suite (971 passed, 1 skipped, 6 deselected) and the compose config, CPU image build and smoke. ubuntu-24.04 (x86) had 2 failures, both SFT golden-hash tests.
- Run https://github.com/SeongHu-Kim/MedQuAD/actions/runs/37923900252: the frozen test tokenizer (D-079) did not change the two x86 failures (same hashes), so it was not the cause.
- Run https://github.com/SeongHu-Kim/MedQuAD/actions/runs/37926168694, temporary golden diagnostic (x86 job 113805525360, arm job 113805525154): identical library versions, tokenizer and embeddings on both runners. The cause is the test fixture's scoring: `HashingEmbedder` gives many exactly tied float32 scores, numpy's default (unstable) `argsort` orders ties differently per CPU, and float32 BLAS results differ in the last bit, which also breaks near-ties differently. With a stable sort the printed top-5 orders were identical on both runners.
- Fix (D-080): the two original golden tests (HashingEmbedder) run on aarch64 only, where all training and evaluation ran; new golden tests use a test-only exact embedder (`tests/training/exact_embedder.py`: distinct integer scores below 2^24) and run on every runner (confirmed on x86 and arm in run 37930186959, D-081).
- On both runners the Keras classifier test is skipped (TensorFlow is not installed in CI), so **CI does not cover the Keras classifier**; it is tested only locally.
- pip-audit (non-blocking) ignores exactly nltk PYSEC-2026-3740 / GHSA-8mgp-746c-j5xp (accepted in D-079; remove when nltk releases a fix) and was green in run 37926168694. The setuptools advisories were fixed by the 83.0.0 pin (confirmed in run 37923900252).

| Job | What it runs | Runners |
|---|---|---|
| `test` | CPU torch from the PyTorch CPU index, plus the lock minus the CUDA stack, then `ruff check`, `ruff format --check`, `mypy src/medquad_qa`, and `pytest -m "not gpu and not real_model and not real_data and not docker and not slow"` | ubuntu-24.04 (x86_64), ubuntu-24.04-arm |
| `compose-and-images` | `docker compose config -q` for the cpu, gpu and monitoring profiles; build `medquad-api:ci-cpu` and `medquad-ui:ci` (no push); `scripts/ops/image_smoke.sh` | ubuntu-24.04-arm |
| `pip-audit` | Dependency audit of the lock, ignoring only the accepted nltk advisory (D-079). Non-blocking (`continue-on-error`). | ubuntu-24.04 |

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
