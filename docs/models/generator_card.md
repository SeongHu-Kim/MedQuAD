# Model card: generator (base and LoRA fine-tuned)

Research prototype for general medical information. Not medical advice, not clinically validated, and no
patient-outcome claims. Answer quality for the four experiment modes is measured by evaluation-safety-engineer, not here.

## Identity

| | Base | Fine-tuned |
|---|---|---|
| `model_version` | `Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554` | `…@cdbee75f…+lora:sft-main-20261006-081347-3f16e30e` |
| Weights | 8.06 GB safetensors; sha256 of every file verified against Hub LFS (`artifacts/models/manifests/base_model.json`) | LoRA adapter, 132 MB, sha256 `fc8d5d08…8011` (`artifacts/models/adapters/sft-main-20261006-081347-3f16e30e/`) |
| Licence | Apache-2.0 (checked on the Hub card at download) | Derived adapter; MedQuAD's licence terms are unverified (D-004 / data card), so use it locally only and do not redistribute |
| Prompt | Closed-book: `rag.prompts.build_closed_book_messages` (`cb-v1+a1f08aaf`). RAG: `build_rag_messages` (owned by retrieval-engineer) | Trained on the same closed-book prompt; served with both |

## Intended use

- The four-way controlled comparison: base, rag, finetuned and finetuned_rag.
- The demo API, which shows a disclaimer.

Out of scope:
- Diagnosis, dosing or treatment decisions.
- Any claim that fine-tuning improves factual accuracy. The training objective only imitates MedQuAD reference answers.

## Inference settings (matched across modes, D-012)

- Greedy decoding, `max_new_tokens` 256, repetition penalty 1.05.
- `max_input_tokens` 3072: a longer prompt raises `GenerationError`. Prompts are never truncated.
- 60 s internal deadline, raising `GenerationTimeoutError`.
- bf16 with SDPA attention on CUDA; fp32 on CPU (smoke only).
- Details: `docs/models/inference.md`.

Measured (`artifacts/models/bench/generator_base_cuda.json`, GB10):

| Measure | Value |
|---|---|
| Load | 5.9 s |
| Single 256-token answer | median 12.9 s (~19.9 tok/s) |
| Batch of 8 | ~150 tok/s aggregate |
| Peak CUDA memory | 8.2 GiB |
| Fine-tuned view | Shares the base weights; the adapter is toggled under a lock |

## Fine-tuning summary

Details: `docs/models/training.md`.

- Method: closed-book LoRA SFT (r16, α32, dropout 0.05, all linear layers), 1 epoch, 790 steps, effective batch 16,
  lr 2e-4 cosine, seed 42. Loss covers assistant tokens only.
- Data: 12,636 TRAIN-split examples. Boilerplate answers are excluded. 369 answers are truncated at a sentence boundary
  with no end-of-turn token.
- Splits: frozen split `split-20261006-2f0fb25ee6d8`. Validation was used only for loss. Test was never read by
  training code. The evaluator's leakage gate passed on the IDs-only record list.
- Runtime and losses: 1.22 h on GB10; validation loss 2.157 → 1.119 on 512 validation examples.
- Tracking: MLflow run `ac6321e7fadd48079f9b307d1146e648` (local SQLite store).
- Reload: verified in a fresh process with real generation on validation questions (`reload_check.json`), then
  promoted to `adapters/CURRENT`.

## Known limitations

- The lower validation loss mostly reflects MedQuAD's answer style. Reload samples show the fine-tuned model writing
  shorter, plainer answers, and at least one confidently states a wrong gene. That is the hallucination risk the RAG
  modes and citation checks are meant to reduce.
- Truncated training examples teach no stopping behaviour for very long answers, which mostly come from CancerGov and GARD.
- English only. The fine-tuning data is a static MedQuAD snapshot, so the content may be out of date.
- Host constraints: torch's Triton op overrides are disabled on this machine (`Python.h` missing), so neither
  `torch.compile` nor static-cache generation is available. Throughput above uses aten fallback kernels.

## Reproduce

```bash
.venv/bin/python scripts/training/download_model.py                                   # verified snapshot + manifest
.venv/bin/python scripts/training/bench_generator.py --variant base --device cuda --out artifacts/models/bench/generator_base_cuda.json
.venv/bin/python scripts/training/train_lora.py --run-kind smoke --max-steps 20
.venv/bin/python scripts/training/train_lora.py --run-kind main --max-steps 790 --expect-prompt-version cb-v1+a1f08aaf
.venv/bin/python scripts/training/verify_adapter.py --run-id <run_id> --promote
.venv/bin/pytest tests/models/test_models_real_generator.py -m real_model             # GPU + CPU real-weights tests
```
Set `TORCH_DISABLE_NATIVE_JIT=1 HF_HUB_OFFLINE=1` for all of them (D-030).
