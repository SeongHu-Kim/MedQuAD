"""Dense retrieval: embeddings stored in Qdrant (embedded local mode or server mode)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from medquad_qa.contracts import RetrievalHit, RetrieverUnavailableError
from medquad_qa.retrieval.corpus import CHUNKER_VERSION, texts_fingerprint
from medquad_qa.retrieval.embedding import Embedder
from medquad_qa.retrieval.manifest import (
    IndexManifest,
    compute_index_version,
    environment_info,
    now_iso,
    resolve_active,
)
from medquad_qa.retrieval.settings import MODE_SUFFIX, RetrievalSettings
from medquad_qa.retrieval.store import CorpusStore, RecordCandidate, ScoredChunk, max_p, to_hits, validate_top_k

UPSERT_BATCH = 512


def qdrant_location(settings: RetrievalSettings) -> dict[str, Any]:
    """Explicit path wins (local mode); else URL (server mode); else local under index_dir."""
    if settings.qdrant_path is not None:
        return {"path": str(settings.qdrant_path)}
    if settings.qdrant_url:
        return {"url": settings.qdrant_url}
    return {"path": str(settings.index_dir / "qdrant_local")}


def open_qdrant(settings: RetrievalSettings, timeout: int = 10) -> Any:
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise RetrieverUnavailableError("qdrant-client not installed (install the 'retrieval' extra)") from exc
    loc = qdrant_location(settings)
    try:
        if "path" in loc:
            Path(loc["path"]).mkdir(parents=True, exist_ok=True)
            return QdrantClient(path=loc["path"])
        client = QdrantClient(url=loc["url"], timeout=timeout, check_compatibility=False)
        client.get_collections()  # fail fast if the server is down
        return client
    except RetrieverUnavailableError:
        raise
    except Exception as exc:
        raise RetrieverUnavailableError(f"qdrant unavailable at {loc}: {type(exc).__name__}") from exc


def dense_definition(settings: RetrievalSettings, embedder: Embedder) -> dict[str, object]:
    return {
        "chunker": CHUNKER_VERSION,
        "chunk_max_words": settings.chunk_max_words,
        "chunk_overlap_words": settings.chunk_overlap_words,
        "embedding_model": embedder.model_id,
        "embedding_revision": embedder.revision,
        "dim": embedder.dim,
        "normalize": True,
        "distance": "cosine",
        "query_prefix": settings.query_prefix,
    }


def build_dense(store: CorpusStore, settings: RetrievalSettings, embedder: Embedder, client: Any) -> IndexManifest:
    """Embed all chunks and (re)create the Qdrant collection named after the content-addressed version."""
    from qdrant_client import models

    t0 = time.perf_counter()
    mode = settings.index_text_mode
    texts = store.index_texts(mode)
    ids = store.chunk_ids()
    texts_sha = texts_fingerprint(texts, ids)
    definition = dense_definition(settings, embedder)
    version = compute_index_version("dense", mode, definition, texts_sha, store.corpus_version)
    if client.collection_exists(version):
        client.delete_collection(version)
    client.create_collection(
        collection_name=version,
        vectors_config=models.VectorParams(size=embedder.dim, distance=models.Distance.COSINE),
    )
    for start in range(0, len(texts), UPSERT_BATCH):
        batch = texts[start : start + UPSERT_BATCH]
        vectors = embedder.encode_documents(batch)
        client.upsert(
            collection_name=version,
            points=[
                models.PointStruct(
                    id=start + j,
                    vector=vectors[j].tolist(),
                    payload={"chunk_id": ids[start + j], "record_id": store.chunks[start + j].record_id},
                )
                for j in range(len(batch))
            ],
            wait=True,
        )
    return IndexManifest(
        index_version=version,
        kind="dense",
        index_text_mode=mode,
        corpus_version=store.corpus_version,
        corpus_sha256=store.corpus_sha256,
        n_records=len(store.records),
        n_chunks=len(ids),
        texts_sha256=texts_sha,
        definition=definition,
        qdrant={"collection": version, "points": len(ids), "location_kind": next(iter(qdrant_location(settings)))},
        build_seconds=round(time.perf_counter() - t0, 3),
        created_at=now_iso(),
        environment=environment_info(),
    )


def verify_dense(manifest: IndexManifest, client: Any, store: CorpusStore | None = None) -> list[str]:
    problems: list[str] = []
    collection = (manifest.qdrant or {}).get("collection", manifest.index_version)
    try:
        if not client.collection_exists(collection):
            return [f"qdrant collection missing: {collection}"]
        count = client.count(collection_name=collection, exact=True).count
    except Exception as exc:
        return [f"qdrant error: {type(exc).__name__}"]
    if count != manifest.n_chunks:
        problems.append(f"point count {count} != manifest n_chunks {manifest.n_chunks}")
    if store is not None:
        if store.corpus_version != manifest.corpus_version:
            problems.append(f"corpus_version {store.corpus_version} != manifest {manifest.corpus_version}")
        texts_sha = texts_fingerprint(store.index_texts(manifest.index_text_mode), store.chunk_ids())
        if texts_sha != manifest.texts_sha256:
            problems.append("indexed texts drifted from the loaded corpus")
        # spot-check payload alignment
        for pid in sorted({0, manifest.n_chunks // 2, manifest.n_chunks - 1}):
            if pid < 0 or pid >= len(store.chunks):
                continue
            pts = client.retrieve(collection_name=collection, ids=[pid], with_payload=True)
            if not pts or pts[0].payload.get("chunk_id") != store.chunks[pid].chunk_id:
                problems.append(f"payload mismatch at point {pid}")
    return problems


class DenseRetriever:
    """``Retriever`` backed by a Qdrant collection of chunk embeddings."""

    def __init__(
        self,
        store: CorpusStore,
        embedder: Embedder,
        client: Any,
        collection: str,
        *,
        mode: str = "answer",
        index_version: str | None = None,
        candidate_k: int = 50,
        collapse_duplicates: bool = False,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.client = client
        self.collection = collection
        self.name = f"dense:{MODE_SUFFIX[mode]}"
        self.corpus_version = store.corpus_version
        self.index_version = index_version
        self.candidate_k = candidate_k
        self.collapse_duplicates = collapse_duplicates

    @classmethod
    def from_settings(
        cls, store: CorpusStore, settings: RetrievalSettings, embedder: Embedder, client: Any
    ) -> DenseRetriever:
        name = f"dense:{settings.mode_suffix}"
        manifest = resolve_active(settings.index_dir, name)
        if manifest.corpus_version != store.corpus_version:
            raise RetrieverUnavailableError(
                f"{name} index built for corpus {manifest.corpus_version}, loaded corpus is {store.corpus_version}"
            )
        if manifest.definition.get("embedding_model") != embedder.model_id or (
            manifest.definition.get("embedding_revision") != embedder.revision
        ):
            raise RetrieverUnavailableError(f"{name}: embedder does not match the index manifest")
        collection = (manifest.qdrant or {}).get("collection", manifest.index_version)
        try:
            exists = client.collection_exists(collection)
        except Exception as exc:
            raise RetrieverUnavailableError(f"qdrant error: {type(exc).__name__}") from exc
        if not exists:
            raise RetrieverUnavailableError(f"qdrant collection missing: {collection} (run the build command)")
        return cls(
            store,
            embedder,
            client,
            collection,
            mode=settings.index_text_mode,
            index_version=manifest.index_version,
            candidate_k=settings.candidate_k,
            collapse_duplicates=settings.collapse_duplicates,
        )

    def search_chunks(self, query: str, k: int) -> list[ScoredChunk]:
        if not query.strip() or k <= 0:
            return []
        vector = self.embedder.encode_queries([query])[0]
        try:
            resp = self.client.query_points(
                collection_name=self.collection, query=np.asarray(vector).tolist(), limit=k, with_payload=False
            )
        except Exception as exc:
            raise RetrieverUnavailableError(f"qdrant query failed: {type(exc).__name__}") from exc
        pts = [(int(p.id), float(p.score)) for p in resp.points]
        pts.sort(key=lambda t: (-t[1], t[0]))
        return [ScoredChunk(i, s) for i, s in pts if 0 <= i < len(self.store.chunks)]

    def search_records(self, query: str, limit: int) -> list[RecordCandidate]:
        depth = max(limit, self.candidate_k) * 4
        return max_p(self.store, self.search_chunks(query, depth), limit, collapse_duplicates=self.collapse_duplicates)

    def retrieve(self, query: str, top_k: int) -> list[RetrievalHit]:
        validate_top_k(top_k)
        return to_hits(self.store, self.search_records(query, top_k), self.name, self.index_version)
