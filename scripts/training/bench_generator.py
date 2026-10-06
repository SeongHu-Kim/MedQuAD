"""Benchmark a generator variant: load time, latency, decode tokens/s, peak memory. Writes one JSON.

Usage:
  .venv/bin/python scripts/training/bench_generator.py --variant base --device cuda \
      --out artifacts/models/bench/generator_base_cuda.json
  .venv/bin/python scripts/training/bench_generator.py --variant base --device cpu --n 1 --max-new-tokens 16 \
      --batch-size 0 --no-long-prompt --out artifacts/models/bench/generator_base_cpu_smoke.json

Prompts are short generic questions written for this benchmark (no dataset text). The long-prompt case pads
the user turn with filler sentences up to ~2,800 tokens to measure prefill near max_input_tokens.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

import torch
import transformers

from medquad_qa.contracts.interfaces import ChatMessage, GenerationResult
from medquad_qa.contracts.qa import GenerationParams
from medquad_qa.models.factory import load_generator
from medquad_qa.models.settings import ModelSettings
from medquad_qa.models.torch_runtime import apply_native_jit_guard

SYSTEM = "You provide general medical information for educational purposes. You do not give personal medical advice."
QUESTIONS = [
    "What is high blood pressure?",
    "What are common symptoms of iron deficiency anemia?",
    "How is type 2 diabetes usually diagnosed?",
    "What can trigger migraine headaches?",
    "What is asthma and how is it managed in general?",
]
FILLER = "This sentence is neutral filler text used only to lengthen the benchmark prompt. "


def _msgs(question: str) -> list[ChatMessage]:
    return [ChatMessage(role="system", content=SYSTEM), ChatMessage(role="user", content=question)]


def _meminfo_available_gib() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return None


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()  # noqa: S603, S607
    except (OSError, subprocess.CalledProcessError):
        return None


def _summary(results: list[GenerationResult]) -> dict[str, Any]:
    lat = [r.latency_ms for r in results]
    tps = [r.completion_tokens / (r.latency_ms / 1000.0) for r in results]
    return {
        "n": len(results),
        "latency_ms_median": median(lat),
        "latency_ms_max": max(lat),
        "completion_tokens": [r.completion_tokens for r in results],
        "prompt_tokens": [r.prompt_tokens for r in results],
        "finish_reasons": [r.finish_reason for r in results],
        "tokens_per_s_median": median(tps),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["base", "finetuned"], default="base")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8, help="0 disables the batch measurement")
    ap.add_argument("--no-long-prompt", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base_settings = ModelSettings.from_env()
    settings = base_settings.model_copy(update={"device": args.device})
    params = GenerationParams(max_new_tokens=args.max_new_tokens)
    mem_before = _meminfo_available_gib()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.monotonic()
    gen = load_generator(args.variant, settings)
    load_s = time.monotonic() - t0

    warm = gen.generate(_msgs("Say hello."), GenerationParams(max_new_tokens=8))
    singles = [gen.generate(_msgs(q), params) for q in (QUESTIONS * 2)[: args.n]]
    report: dict[str, Any] = {
        "model_version": gen.model_version,
        "variant": args.variant,
        "device": str(gen.backend.device),
        "dtype": str(next(gen.backend.model.parameters()).dtype),
        "generation": {
            "params": params.model_dump(),
            "repetition_penalty": gen.config.repetition_penalty,
            "max_input_tokens": gen.config.max_input_tokens,
        },
        "load_s": load_s,
        "warmup_latency_ms": warm.latency_ms,
        "single": _summary(singles),
        "samples": [{"question": q, "answer_head": r.text[:300]} for q, r in zip(QUESTIONS, singles, strict=False)],
    }

    if not args.no_long_prompt:
        n_filler = 1
        while gen.count_tokens(_msgs(FILLER * n_filler + QUESTIONS[0])) < 2800:
            n_filler += 20
        long_msgs = _msgs(FILLER * n_filler + QUESTIONS[0])
        report["long_prompt"] = _summary([gen.generate(long_msgs, GenerationParams(max_new_tokens=64))])

    if args.batch_size > 0:
        batch = [_msgs(q) for q in (QUESTIONS * args.batch_size)[: args.batch_size]]
        t1 = time.monotonic()
        results = gen.generate_batch(batch, params)
        wall = time.monotonic() - t1
        total = sum(r.completion_tokens for r in results)
        report["batch"] = {
            "batch_size": args.batch_size,
            "wall_s": wall,
            "completion_tokens_total": total,
            "aggregate_tokens_per_s": total / wall,
            "finish_reasons": [r.finish_reason for r in results],
        }

    report["memory"] = {
        "cuda_max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
        "cuda_max_reserved_gib": torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else None,
        "process_max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        "system_mem_available_gib_before": mem_before,
        "system_mem_available_gib_after": _meminfo_available_gib(),
        "note": "GB10 unified memory: nvidia-smi reports N/A; CUDA numbers come from the torch caching allocator.",
    }
    report["env"] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "machine": platform.machine(),
        "python": platform.python_version(),
        "git_sha": _git_sha(),
        "torch_native_triton_ops": apply_native_jit_guard(),
    }
    report["created_at_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("model_version", "device", "load_s", "single", "memory")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
