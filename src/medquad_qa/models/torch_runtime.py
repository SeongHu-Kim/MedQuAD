"""Torch runtime guard for this host.

torch 2.14 routes some aten ops (e.g. the batched outer product used by Qwen3 rotary embeddings) to Triton
kernels in ``torch._native``. Triton JIT-compiles a small C driver that needs ``Python.h``; this host has no
python3.12-dev headers, so the first such op fails with ``fatal error: Python.h: No such file or directory``.
When the header is missing we deregister those optional overrides and torch falls back to its built-in aten
kernels. Setting ``TORCH_DISABLE_NATIVE_JIT=1`` before torch is imported has the same effect.
"""

from __future__ import annotations

import logging
import os
import sysconfig
from pathlib import Path

log = logging.getLogger(__name__)
_STATE: str | None = None


def python_headers_available() -> bool:
    include = sysconfig.get_paths().get("include")
    return include is not None and (Path(include) / "Python.h").is_file()


def apply_native_jit_guard() -> str:
    """Idempotent. Returns a short status string recorded in bench/run manifests."""
    global _STATE
    if _STATE is not None:
        return _STATE
    if os.environ.get("TORCH_DISABLE_NATIVE_JIT") == "1":
        _STATE = "disabled (TORCH_DISABLE_NATIVE_JIT=1)"
    elif python_headers_available():
        _STATE = "enabled"
    else:
        try:
            from torch._native import triton_utils

            triton_utils.deregister_op_overrides()
            _STATE = "disabled (Python.h missing; triton op overrides deregistered)"
            log.warning("Python.h not found: torch native Triton op overrides disabled, using aten kernels")
        except Exception as exc:  # private torch API; never fail model loading because of it
            _STATE = f"unchanged (deregister failed: {type(exc).__name__})"
    return _STATE
