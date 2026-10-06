# LoRA supervised fine-tuning (model-engineer)

Closed-book SFT of `Qwen/Qwen3-4B-Instruct-2507@cdbee75…` on the TRAIN split only (D-013). The run manifest is the
source of truth for every number: `artifacts/models/runs/<run_id>/run_manifest.json`.

## Data

- Input: `data/processed/exports/records_train.jsonl` from the frozen split `split-20261006-2f0fb25ee6d8`. The sha256 of
  all three exports is verified through `medquad_qa.data.corpus.load_split` before training. Validation is used only
  for validation loss. Test is never read by training code.
- Leakage guards:
  - `check_split_isolation` aborts if any train record or split group appears in the validation or test exports.
  - The evaluator's `scripts/evaluation/check_protected_leakage.py` passed on `sft_record_ids.jsonl` (IDs only).
- Prompt: `medquad_qa.rag.prompts.build_closed_book_messages` is the exact system and user format the pipeline uses
  for `base` and `finetuned`. Its version (`CLOSED_BOOK_PROMPT_VERSION`) is recorded in the run and adapter manifests.
  `train_lora.py --expect-prompt-version` aborts on a mismatch.
- Exclusions: records flagged `boilerplate_answer` are not used as targets (378 of 13,014 train records).

## Example construction (`training/sft_data.py`)

- Sequence = chat template(system, user) + `<|im_start|>assistant\n` + answer + `<|im_end|>`. The labels mask every prompt token
  (−100), so the loss covers assistant tokens only. A check on 50 records found the builder's token ids identical to
  the Qwen tokenizer's own rendering of the full conversation (no `<think>` block is inserted).
- Max length 1024. Answers that do not fit are cut at the last sentence boundary that fits, and the end-of-turn token is
  **not** appended, so the model never learns to stop where an answer was cut. Counts per source and dropped answer
  tokens are written to `sft_build_train.json`.
  - Effect: 369 of 12,636 examples (2.9%) are truncated, dropping about 8% of answer tokens, mostly in long CancerGov
    and GARD answers.
  - The model sees no end-of-turn token for those examples, and can produce only a shorter, cut-down version of them.

## Training (`training/lora.py`, `scripts/training/train_lora.py`)

| Setting | Value (`configs/training/lora_sft.yaml`) |
|---|---|
| LoRA | r 16, α 32, dropout 0.05, all linear layers (252 modules), 33 M trainable params |
| Optimiser | AdamW (torch), lr 2e-4, cosine, warmup 3% of steps, weight decay 0, grad-norm clip 1.0 |
| Batch | 4 per device × 4 accumulation = 16 sequences per step; length-grouped sampling |
| Precision | bf16 autocast, fp32 LoRA weights and optimiser state, gradient checkpointing |
| Schedule | 1 epoch = 790 steps; hard wall-clock cap 2.5 h (status `partial_time_cap` if hit) |
| Seed | 42 (model init, data order, sampler) |
| Tracking | MLflow (server if reachable, else `sqlite:///artifacts/models/mlruns-local/mlflow.db`, D-026) + `metrics.jsonl` + `loss_curve.png` |

GB10 / torch 2.14 specifics:
- **Adapter input casting:** with fp32 LoRA weights, PEFT casts every adapter input to fp32. Profiling one 4×512
  micro-batch measured 3.85 s with casting and 1.51 s without. Training disables that cast under bf16 autocast, so the
  adapter matmuls run in bf16 while weights and optimiser stay fp32. After 20 steps the validation loss was 1.28102
  with casting and 1.28168 without. Mean step time fell from 18.3 s to 8.9 s (smoke runs
  `sft-smoke-20261006-074026-*` and `sft-smoke-20261006-080622-*`).
- **Aten fallback kernels:** torch's Triton op overrides are disabled because `Python.h` is missing (D-030), so
  training runs on aten fallback kernels. `torch.compile` is not used.
- **Contention:** a second GPU job roughly doubled the step time in one smoke run (20.0 s/step), so the main run was
  scheduled in a GPU-quiet window.

## Verification

`scripts/training/verify_adapter.py --run-id <id> --promote` runs in a fresh process. It does four things:
1. Loads base and fine-tuned views through `load_generator`, with shared weights.
2. Generates greedy answers to three generic questions written for this check.
3. Checks that the fine-tuned `model_version` carries the run id, the backend is shared, outputs are non-empty, and at
   least one output differs from base.
4. Writes `reload_check.json`. Only if every check passes does it write `adapters/CURRENT`.
