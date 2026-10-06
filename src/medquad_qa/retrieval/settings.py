"""Retrieval settings (plain pydantic, read from MEDQUAD_* environment variables; D-024)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RetrieverKind = Literal["bm25", "dense", "dense_fallback", "hybrid"]
IndexTextMode = Literal["answer", "question_answer"]

#: Short suffix used in retriever names (D-024): ``bm25:answer``, ``hybrid_rrf:qa``.
MODE_SUFFIX: dict[str, str] = {"answer": "answer", "question_answer": "qa"}

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_EMBEDDING_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"  # MIT licence, verified 2026-10-06
DEFAULT_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANKER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"  # Apache-2.0, verified 2026-10-06


class RetrievalSettings(BaseModel):
    """All knobs that affect retrieval output. Fields marked as index-defining feed ``index_version``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    retriever: RetrieverKind = "hybrid"
    index_text_mode: IndexTextMode = "answer"
    corpus_path: Path = Path("data/processed/corpus.jsonl")
    corpus_manifest_path: Path = Path("data/manifests/corpus_manifest.json")
    index_dir: Path = Path("artifacts/indexes")
    qdrant_url: str | None = None
    qdrant_path: Path | None = None
    # chunking (index-defining)
    chunk_max_words: int = Field(default=200, ge=20)
    chunk_overlap_words: int = Field(default=40, ge=0)
    # bm25 (index-defining)
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    use_stemmer: bool = True
    # dense (index-defining)
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_revision: str | None = DEFAULT_EMBEDDING_REVISION
    query_prefix: str = DEFAULT_QUERY_PREFIX
    embedding_batch_size: int = Field(default=64, ge=1)
    device: str = "auto"
    # query time
    candidate_k: int = Field(default=50, ge=1, description="Per-retriever depth fused by RRF.")
    rrf_k: int = Field(default=60, ge=1)
    collapse_duplicates: bool = False
    reranker_model: str | None = None
    reranker_revision: str | None = DEFAULT_RERANKER_REVISION
    rerank_depth: int = Field(default=20, ge=1)

    @classmethod
    def from_env(cls, **overrides: object) -> RetrievalSettings:
        env = os.environ
        values: dict[str, object] = {}
        if env.get("MEDQUAD_RETRIEVER"):
            values["retriever"] = env["MEDQUAD_RETRIEVER"].strip()
        if env.get("MEDQUAD_INDEX_TEXT_MODE"):
            values["index_text_mode"] = env["MEDQUAD_INDEX_TEXT_MODE"].strip()
        if env.get("MEDQUAD_CORPUS_PATH"):
            values["corpus_path"] = Path(env["MEDQUAD_CORPUS_PATH"].strip())
        if env.get("MEDQUAD_CORPUS_MANIFEST"):
            values["corpus_manifest_path"] = Path(env["MEDQUAD_CORPUS_MANIFEST"].strip())
        if env.get("MEDQUAD_INDEX_DIR"):
            values["index_dir"] = Path(env["MEDQUAD_INDEX_DIR"].strip())
        if env.get("MEDQUAD_QDRANT_PATH"):
            values["qdrant_path"] = Path(env["MEDQUAD_QDRANT_PATH"].strip())
        elif env.get("MEDQUAD_QDRANT_URL"):
            values["qdrant_url"] = env["MEDQUAD_QDRANT_URL"].strip()
        if env.get("MEDQUAD_EMBEDDING_REVISION"):
            values["embedding_revision"] = env["MEDQUAD_EMBEDDING_REVISION"].strip()
        if env.get("MEDQUAD_DEVICE"):
            values["device"] = env["MEDQUAD_DEVICE"].strip()
        if env.get("MEDQUAD_RERANKER_MODEL"):
            values["reranker_model"] = env["MEDQUAD_RERANKER_MODEL"].strip()
        values.update(overrides)
        return cls.model_validate(values)

    @property
    def mode_suffix(self) -> str:
        return MODE_SUFFIX[self.index_text_mode]
