"""Platform-independent test embedder: exact integer scores with a unique per-document tie-breaker (D-080).

HashingEmbedder's normalised float32 vectors give many exact ties and last-bit BLAS differences, so numpy's
(unstable, CPU-specific) argsort ordered the SFT fixture's candidates differently on x86 and aarch64. Here every
query·document score is ``SCALE * token_overlap + (n_docs - 1 - doc_index)``: non-negative integers below 2**24,
so float32 sums are exact in any order, and no two documents of one call share a score. Among equal overlaps the
earlier document ranks higher, as a stable sort would.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

SCALE = 1024  # > number of documents per call, so the tie-breaker never outweighs one unit of overlap
MAX_QUERY_TOKENS = 32
MAX_DOC_TOKENS = 500  # SCALE * MAX_QUERY_TOKENS * MAX_DOC_TOKENS + SCALE < 2**24


class ExactTieBreakEmbedder:
    """Offline test double; not a semantic model. Rankings differ from HashingEmbedder's (unnormalised counts)."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _counts(self, texts: list[str], max_tokens: int) -> np.ndarray:
        out = np.zeros((len(texts), self.dim + 1), dtype=np.float32)
        for row, text in enumerate(texts):
            toks = re.findall(r"[a-z0-9]+", text.lower())
            if len(toks) > max_tokens:
                raise ValueError(f"text has {len(toks)} tokens > {max_tokens}: scores could exceed 2**24")
            for tok in toks:
                h = int.from_bytes(hashlib.sha256(tok.encode()).digest()[:4], "little")
                out[row, h % self.dim] += 1.0
        return out

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        if len(texts) >= SCALE:
            raise ValueError(f"{len(texts)} documents >= {SCALE}: tie-breaker would collide with overlap units")
        out = self._counts(texts, MAX_DOC_TOKENS)
        out[:, self.dim] = np.arange(len(texts) - 1, -1, -1, dtype=np.float32)
        return out

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        out = self._counts(texts, MAX_QUERY_TOKENS) * SCALE
        out[:, self.dim] = 1.0
        return out
