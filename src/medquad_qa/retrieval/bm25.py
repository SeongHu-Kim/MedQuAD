"""BM25 lexical retrieval (bm25s) over answer chunks; the CPU-only, torch-free fallback retriever."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from medquad_qa.contracts import RetrievalHit, RetrieverUnavailableError
from medquad_qa.retrieval.corpus import CHUNKER_VERSION, texts_fingerprint
from medquad_qa.retrieval.manifest import (
    IndexManifest,
    compute_index_version,
    environment_info,
    file_hashes,
    now_iso,
    resolve_active,
    verify_files,
)
from medquad_qa.retrieval.settings import MODE_SUFFIX, RetrievalSettings
from medquad_qa.retrieval.store import CorpusStore, RecordCandidate, ScoredChunk, max_p, to_hits, validate_top_k
from medquad_qa.retrieval.tokenize import tokenize, tokenizer_id


class BM25ChunkIndex:
    """bm25s index with a deterministic (sorted) vocabulary; row i == store.chunks[i]."""

    def __init__(self, model: object, chunk_ids: list[str], stem: bool) -> None:
        self._model = model
        self.chunk_ids = chunk_ids
        self.stem = stem

    @classmethod
    def build(cls, texts: list[str], chunk_ids: list[str], *, k1: float, b: float, stem: bool) -> BM25ChunkIndex:
        import bm25s
        from bm25s.tokenization import Tokenized

        token_lists = [tokenize(t, stem=stem) for t in texts]
        vocab = {tok: i for i, tok in enumerate(sorted({t for toks in token_lists for t in toks}))}
        ids = [[vocab[t] for t in toks] for toks in token_lists]
        model = bm25s.BM25(k1=k1, b=b)
        model.index(Tokenized(ids=ids, vocab=vocab), show_progress=False)
        return cls(model, list(chunk_ids), stem)

    def save(self, path: Path) -> None:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
        self._model.save(str(path))  # type: ignore[attr-defined]
        (path / "chunk_ids.json").write_text(json.dumps(self.chunk_ids) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path, stem: bool) -> BM25ChunkIndex:
        import bm25s

        if not (path / "chunk_ids.json").is_file():
            raise RetrieverUnavailableError(f"bm25 index files missing under {path}")
        model = bm25s.BM25.load(str(path))
        chunk_ids = json.loads((path / "chunk_ids.json").read_text(encoding="utf-8"))
        return cls(model, chunk_ids, stem)

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        """Top-``k`` chunks with score > 0; ties broken by chunk position (deterministic)."""
        tokens = tokenize(query, stem=self.stem)
        if not tokens or k <= 0:
            return []
        scores = np.asarray(self._model.get_scores(tokens), dtype=np.float64)  # type: ignore[attr-defined]
        positive = np.flatnonzero(scores > 0.0)
        if positive.size == 0:
            return []
        order = positive[np.lexsort((positive, -scores[positive]))][:k]
        return [ScoredChunk(int(i), float(scores[i])) for i in order]


def bm25_definition(settings: RetrievalSettings) -> dict[str, object]:
    return {
        "chunker": CHUNKER_VERSION,
        "chunk_max_words": settings.chunk_max_words,
        "chunk_overlap_words": settings.chunk_overlap_words,
        "tokenizer": tokenizer_id(settings.use_stemmer),
        "k1": settings.bm25_k1,
        "b": settings.bm25_b,
        "method": "lucene",
    }


def build_bm25(store: CorpusStore, settings: RetrievalSettings) -> IndexManifest:
    """Build (or reuse, if content-identical) the BM25 index and return its manifest (not yet activated)."""
    t0 = time.perf_counter()
    mode = settings.index_text_mode
    texts = store.index_texts(mode)
    ids = store.chunk_ids()
    texts_sha = texts_fingerprint(texts, ids)
    definition = bm25_definition(settings)
    version = compute_index_version("bm25", mode, definition, texts_sha, store.corpus_version)
    out_dir = settings.index_dir / "bm25" / version
    index = BM25ChunkIndex.build(texts, ids, k1=settings.bm25_k1, b=settings.bm25_b, stem=settings.use_stemmer)
    index.save(out_dir)
    return IndexManifest(
        index_version=version,
        kind="bm25",
        index_text_mode=mode,
        corpus_version=store.corpus_version,
        corpus_sha256=store.corpus_sha256,
        n_records=len(store.records),
        n_chunks=len(ids),
        texts_sha256=texts_sha,
        definition=definition,
        files=file_hashes(out_dir, settings.index_dir),
        build_seconds=round(time.perf_counter() - t0, 3),
        created_at=now_iso(),
        environment=environment_info(),
    )


class BM25Retriever:
    """``Retriever`` over a built BM25 index. Lexical-only; needs neither torch nor Qdrant."""

    def __init__(
        self,
        store: CorpusStore,
        index: BM25ChunkIndex,
        *,
        mode: str = "answer",
        index_version: str | None = None,
        candidate_k: int = 50,
        collapse_duplicates: bool = False,
    ) -> None:
        if index.chunk_ids != store.chunk_ids():
            raise RetrieverUnavailableError("bm25 index does not match the loaded corpus chunks (rebuild the index)")
        self.store = store
        self.index = index
        self.name = f"bm25:{MODE_SUFFIX[mode]}"
        self.corpus_version = store.corpus_version
        self.index_version = index_version
        self.candidate_k = candidate_k
        self.collapse_duplicates = collapse_duplicates

    @classmethod
    def in_memory(cls, store: CorpusStore, settings: RetrievalSettings | None = None) -> BM25Retriever:
        """Build an unpersisted index (tests, small fixtures)."""
        s = settings or RetrievalSettings()
        texts = store.index_texts(s.index_text_mode)
        index = BM25ChunkIndex.build(texts, store.chunk_ids(), k1=s.bm25_k1, b=s.bm25_b, stem=s.use_stemmer)
        return cls(
            store, index, mode=s.index_text_mode, candidate_k=s.candidate_k, collapse_duplicates=s.collapse_duplicates
        )

    @classmethod
    def from_settings(cls, store: CorpusStore, settings: RetrievalSettings) -> BM25Retriever:
        name = f"bm25:{settings.mode_suffix}"
        manifest = resolve_active(settings.index_dir, name)
        if manifest.corpus_version != store.corpus_version:
            raise RetrieverUnavailableError(
                f"{name} index built for corpus {manifest.corpus_version}, loaded corpus is {store.corpus_version}"
            )
        problems = verify_files(manifest, settings.index_dir)
        if problems:
            raise RetrieverUnavailableError(f"{name} index drift: {problems[:3]}")
        index = BM25ChunkIndex.load(settings.index_dir / "bm25" / manifest.index_version, settings.use_stemmer)
        return cls(
            store,
            index,
            mode=settings.index_text_mode,
            index_version=manifest.index_version,
            candidate_k=settings.candidate_k,
            collapse_duplicates=settings.collapse_duplicates,
        )

    def search_records(self, query: str, limit: int) -> list[RecordCandidate]:
        # Over-fetch chunks so that MaxP still yields ``limit`` distinct records.
        depth = max(limit, self.candidate_k) * 4
        return max_p(self.store, self.index.search(query, depth), limit, collapse_duplicates=self.collapse_duplicates)

    def retrieve(self, query: str, top_k: int) -> list[RetrievalHit]:
        validate_top_k(top_k)
        return to_hits(self.store, self.search_records(query, top_k), self.name, self.index_version)
