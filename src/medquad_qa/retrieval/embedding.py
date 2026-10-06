"""Sentence embedders: the real BGE model (sentence-transformers) and a deterministic offline test double."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from medquad_qa.contracts import ArtifactUnavailableError


@runtime_checkable
class Embedder(Protocol):
    model_id: str
    revision: str
    dim: int

    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    def encode_queries(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic bag-of-hashed-tokens embedder for OFFLINE TESTS ONLY (not a semantic model)."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.model_id = f"test-hashing-embedder-{dim}"
        self.revision = "v1"

    def _encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                h = int.from_bytes(hashlib.sha256(tok.encode()).digest()[:4], "little")
                out[row, h % self.dim] += 1.0
            norm = float(np.linalg.norm(out[row]))
            if norm > 0:
                out[row] /= norm
        return out

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)


#: Only the files sentence-transformers needs (skips pytorch_model.bin / onnx / openvino duplicates).
ALLOW_PATTERNS = ["*.json", "*.txt", "model.safetensors", "1_Pooling/*", "README.md", "sentencepiece.bpe.model"]


def resolve_model_snapshot(model_id: str, revision: str | None, *, allow_download: bool = False) -> tuple[Path, str]:
    """Find (or, if ``allow_download``, download) a model snapshot; return (local_path, commit_sha).

    Serving and query paths never download (``allow_download=False``): a missing model is an
    ArtifactUnavailableError. Only the explicit ``build`` command downloads. The commit sha is
    what manifests pin.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - ml extra missing
        raise ArtifactUnavailableError("huggingface_hub not installed (install the 'ml' extra)") from exc
    try:
        local = Path(
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                allow_patterns=ALLOW_PATTERNS,
                local_files_only=not allow_download,
            )
        )
    except Exception as exc:  # network/cache errors of many kinds
        raise ArtifactUnavailableError(f"embedding model {model_id}@{revision or 'main'} unavailable: {exc}") from exc
    commit = local.name if re.fullmatch(r"[0-9a-f]{40}", local.name) else (revision or "unknown")
    return local, commit


class SentenceTransformerEmbedder:
    """BGE-style bi-encoder: L2-normalised embeddings, instruction prefix on queries only."""

    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        *,
        query_prefix: str = "",
        device: str = "auto",
        batch_size: int = 64,
        allow_download: bool = False,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ArtifactUnavailableError("sentence-transformers not installed (install the 'ml' extra)") from exc
        local, commit = resolve_model_snapshot(model_id, revision, allow_download=allow_download)
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:  # pragma: no cover
                device = "cpu"
        self._model = SentenceTransformer(str(local), device=device)
        self.model_id = model_id
        self.revision = commit
        self.device = device
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        dim = self._model.get_sentence_embedding_dimension()
        if dim is None:
            raise ArtifactUnavailableError(f"cannot determine embedding size for {model_id}")
        self.dim = int(dim)

    def _encode(self, texts: list[str]) -> np.ndarray:
        emb = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(emb, dtype=np.float32)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.query_prefix + t for t in texts])
