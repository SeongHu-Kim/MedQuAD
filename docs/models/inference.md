# Generator inference (model-engineer)

Status: M2 implemented. Measured numbers live only in `artifacts/models/bench/*.json` (see "Evidence" below).

## Model

| Item | Value |
|---|---|
| Base model | `Qwen/Qwen3-4B-Instruct-2507` (non-thinking instruct; ChatML template from the tokenizer) |
| Revision | `cdbee75f17c01a7cc42f958dc650907174af0554` (pinned; D-011) |
| Licence | Apache-2.0, checked against the Hub card at download (`artifacts/models/manifests/base_model.json`) |
| Files | safetensors, config, tokenizer files only; sha256 checked against Hub LFS metadata |
| `model_version` | `Qwen/Qwen3-4B-Instruct-2507@cdbee75…` (base) or `…+lora:<run_id>` (fine-tuned) |

Download (once, ~8 GB, into the default HF cache `~/.cache/huggingface`):

```bash
.venv/bin/python scripts/training/download_model.py
```

Loading never downloads (`local_files_only=True`) unless `MEDQUAD_ALLOW_MODEL_DOWNLOAD=true`.

## Matched generation settings (D-012)

`configs/models/generator.yaml` (mirrors `GeneratorConfig` defaults; a test keeps them equal):
greedy decoding (`temperature=0.0`), `max_new_tokens=256`, `repetition_penalty=1.05`, `max_input_tokens=3072`,
internal deadline `timeout_s=60`, bf16 + SDPA on CUDA, fp32 on CPU (smoke path only).

## API (`medquad_qa.models`)

```python
from medquad_qa.models import load_generator, generator_status, ModelSettings, FakeGenerator

gen = load_generator("base")            # or "finetuned"; settings default to ModelSettings.from_env()
gen.generate(messages, params)          # -> GenerationResult
gen.generate_batch(batch, params)       # -> list[GenerationResult]  (offline evaluation)
gen.count_tokens(messages)              # exact prompt tokens incl. chat template + generation prompt (D-022)
generator_status("finetuned")           # ComponentStatus from files only; never raises, loads nothing
```

Behaviour and errors:

| Condition | Result |
|---|---|
| Prompt tokens > `max_input_tokens` | `GenerationError` (never silently truncated) |
| Wall clock > `timeout_s` (including waiting for the generator lock) | `GenerationTimeoutError` |
| Any runtime failure inside `model.generate` (CUDA, OOM, kernel) | `GenerationError` with the original exception chained |
| Base weights missing / unloadable | `ArtifactUnavailableError` at load |
| `finetuned` with no adapter, an adapter for another base, or no `run_id` | `ModeUnavailableError("finetuned")` |
| EOS emitted | `finish_reason="stop"`; `completion_tokens` includes the EOS token |
| `max_new_tokens` reached | `finish_reason="length"` |

`generate_batch` left-pads, uses a deadline of `timeout_s × len(batch)`, and reports the batch wall time as each
item's `latency_ms`. Greedy batch outputs can differ slightly from single calls on GPU (padding/kernel choice).
`latency_ms` is wall time from the call start, including any wait for the lock.

Sampling (`temperature>0`) uses `top_p` from the request and disables top-k; `seed` seeds torch inside the lock.

## Shared weights for base and fine-tuned modes

One `GeneratorBackend` per (model, revision, device, dtype) is cached in-process. Loading `finetuned` attaches the
LoRA adapter to that same model (`PeftModel`), so both modes cost one copy of the weights. Every generation holds
the backend lock; the base view runs inside `disable_adapter()`, the fine-tuned view activates its adapter.
Tests verify that base outputs and logits are unchanged after an adapter is attached, and that interleaved
threads give the same outputs as sequential calls. Consequence for serving: generation is serialised per
process (the API's global semaphore matches this).

Adapter resolution: `MEDQUAD_ADAPTER_DIR`, else `<MEDQUAD_MODEL_DIR>/adapters/CURRENT` (one line: run id) →
`<MEDQUAD_MODEL_DIR>/adapters/<run_id>/`, which must contain `adapter_config.json`, the adapter weights and
`adapter_manifest.json` (`run_id`, `base_model_id`, `base_revision`, …).

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `MEDQUAD_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `MEDQUAD_MODEL_DIR` | `artifacts/models` | adapters and model manifests |
| `MEDQUAD_ADAPTER_DIR` | unset | explicit adapter directory (overrides `CURRENT`) |
| `MEDQUAD_GENERATOR_CONFIG` | `configs/models/generator.yaml` if present | generator YAML |
| `MEDQUAD_ALLOW_MODEL_DOWNLOAD` | `false` | allow Hub downloads at load time |
| `HF_HOME` / `HF_HUB_OFFLINE` | HF defaults | cache location / offline mode |

## Tests

- Offline (in `make test`): `tests/models/test_models_*.py` except `test_models_real_generator.py`. They use a tiny
  random-init Qwen3 with a ChatML tokenizer built offline (`medquad_qa.models.testing`).
- Real weights: `.venv/bin/pytest tests/models/test_models_real_generator.py -m real_model` (GPU test marked `gpu`;
  CPU smoke marked `slow`).

## Host-specific notes (GB10, torch 2.14.1+cu130)

- **Triton / `Python.h`:** torch 2.14 routes some aten ops (e.g. the rotary-embedding outer product) to Triton
  kernels that JIT-compile a C driver against `Python.h`. This host has no python3.12-dev headers, so the first
  CUDA forward failed with `fatal error: Python.h: No such file or directory`. `models.torch_runtime.apply_native_jit_guard()`
  (called on every model load) deregisters those optional overrides and torch uses its aten kernels.
  `TORCH_DISABLE_NATIVE_JIT=1` set before importing torch has the same effect. `torch.compile` and static-cache
  generation (which compiles) are unavailable for the same reason; nothing here uses them.
- **Load time:** with memory-mapped safetensors, copying the weights to the GPU ran at ~150 MB/s (page faults on
  untouched pages), so a load took ~50 s. Loading with `disable_mmap=True` takes ~5–6 s. Process RSS peaks at
  ~12 GiB during load because the files are read into memory first.

## Measured (bench JSON is the source of truth)

From `artifacts/models/bench/generator_base_cuda.json` (bf16, CUDA, greedy, max_new_tokens 256, repetition
penalty 1.05; 5 single prompts + one ~3,050-token prompt + one batch of 8):

| Metric | Value |
|---|---|
| Load (`load_generator("base")`) | 5.9 s |
| Single request, 256 new tokens | median 12.9 s (~19.9 tokens/s) |
| ~3,050-token prompt, 64 new tokens | 4.1 s |
| Batch of 8 (`generate_batch`) | 13.5 s wall, ~150 tokens/s aggregate |
| Peak CUDA allocated / reserved | 8.2 / 8.5 GiB |
| Peak process RSS | 12.1 GiB |

Consequences: at ~20 tokens/s a single request with `max_new_tokens` above ~1,000 would exceed the 60 s deadline
and raise `GenerationTimeoutError`; offline evaluation should use `generate_batch`.
CPU fp32 smoke (`generator_base_cpu_smoke.json`): load 5.7 s, ~1.3 tokens/s, RSS 23.7 GiB. Usable for smoke tests only.

## Evidence

- `artifacts/models/manifests/base_model.json`: revision, licence, per-file sha256.
- `artifacts/models/bench/generator_base_cuda.json`: load time, latency, tokens/s, peak memory (bf16, CUDA).
- `artifacts/models/bench/generator_base_cpu_smoke.json`: fp32 CPU smoke.
