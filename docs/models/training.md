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

## Main run results (M4)

Run `sft-main-20261006-081347-3f16e30e`. Every value below is from
`artifacts/models/runs/sft-main-20261006-081347-3f16e30e/run_manifest.json`, `metrics.jsonl` and `reload_check.json`.
The launch command, with `TORCH_DISABLE_NATIVE_JIT=1 HF_HUB_OFFLINE=1` set:

```bash
.venv/bin/python scripts/training/train_lora.py --run-kind main --max-steps 790 --expect-prompt-version cb-v1+a1f08aaf
```

| Item | Value |
|---|---|
| Status | `completed`: 790/790 steps, 1.0 epoch, time cap not hit |
| Training runtime | 4,399 s (1.22 h); 5.54 s/step mean; whole script 4,538 s |
| Train loss (assistant tokens) | step 1: 1.510; step 100: 1.070; step 400: 0.975; step 790: 1.014; mean over the run 1.115 |
| Validation loss (512 validation examples) | 2.157 before training (equal to base, since LoRA B starts at 0) → **1.119** after |
| Peak CUDA memory | 15.79 GiB allocated / 17.60 GiB reserved |
| Trainable parameters | 33,030,144 of 4,055,498,240 (0.81%) |
| Prompt version | `cb-v1+a1f08aaf` (asserted before and after training) |
| Data | split `split-20261006-2f0fb25ee6d8`, corpus `medquad-1.0.0-86e384302357`, export sha256s verified |
| Adapter | `artifacts/models/adapters/sft-main-20261006-081347-3f16e30e/`; `adapter_model.safetensors` sha256 `fc8d5d08d0c4654cc306a65e5dd7af372b21a68e367fedcf5d6508ca8d808011` |
| MLflow | Run `ac6321e7fadd48079f9b307d1146e648`, experiment `medquad-sft`. It was logged live to the offline store `sqlite:///artifacts/models/mlruns-local/mlflow.db` because the MLflow server was unreachable. It holds 80 loss points plus the manifest, loss curve and record-ID artifacts. |
| Kernel warnings | None from CUDA or Triton. The only warnings are PEFT's "Could not find a config file in Qwen/Qwen3-4B-Instruct-2507" at adapter saves (benign: vocabulary unchanged). Triton op overrides were disabled via `TORCH_DISABLE_NATIVE_JIT=1`. |
| Environment | torch 2.14.1+cu130, transformers 5.18.0, peft 0.21.2, NVIDIA GB10 aarch64, Python 3.12.3 |

Loss curve: `artifacts/models/runs/sft-main-20261006-081347-3f16e30e/loss_curve.png`.

Reading the validation loss:
- It measures next-token cross-entropy on MedQuAD-style reference answers. Much of the drop from 2.16 to 1.12 comes
  from learning the corpus's answer style and format. It does not show that answers are more correct.
- Answer quality is measured separately by evaluation-safety-engineer: rubric, NLI support, the four-mode comparison,
  and the memorisation probe.

### Reload and generation check

The command ran in a fresh process:

```bash
.venv/bin/python scripts/training/verify_adapter.py --run-id sft-main-20261006-081347-3f16e30e --promote
```

- It used the first 5 validation questions by record_id (no dev or test items), greedy decoding and 128 new tokens.
- All checks passed:
  - The adapter sha256 matches the run manifest.
  - `model_version` ends with `+lora:sft-main-20261006-081347-3f16e30e`.
  - Base and fine-tuned share one backend.
  - All outputs are non-empty.
  - Fine-tuned output differs from base on 5 of 5 questions.
- The run was then promoted (`adapters/CURRENT`). Only IDs, token counts and checks go in the tracked `reload_check.json`. Text
  samples are in the gitignored `reload_samples.jsonl`.
- What the samples show (an observation, not a correctness claim): the fine-tuned model adopts the MedQuAD
  reference style, shorter and plainer, sometimes opening with "Summary :". At least one sample states a wrong gene
  confidently, so fine-tuning is not evidence of factual improvement.

Fine-tuned `model_version`:
`Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554+lora:sft-main-20261006-081347-3f16e30e`.
