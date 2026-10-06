"""Build the configured retriever from settings, with per-component readiness status."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from medquad_qa.contracts import ArtifactUnavailableError, ComponentStatus, Retriever, RetrieverUnavailableError
from medquad_qa.retrieval.bm25 import BM25Retriever
from medquad_qa.retrieval.dense import DenseRetriever, open_qdrant
from medquad_qa.retrieval.embedding import Embedder, SentenceTransformerEmbedder
from medquad_qa.retrieval.hybrid import CrossEncoderReranker, HybridRetriever
from medquad_qa.retrieval.manifest import resolve_active
from medquad_qa.retrieval.settings import RetrievalSettings
from medquad_qa.retrieval.store import CorpusStore

log = logging.getLogger(__name__)


@dataclass
class RetrievalBundle:
    retriever: Retriever | None
    store: CorpusStore | None
    statuses: list[ComponentStatus] = field(default_factory=list)


def load_store(settings: RetrievalSettings) -> CorpusStore:
    return CorpusStore.from_files(
        settings.corpus_path,
        settings.corpus_manifest_path,
        max_words=settings.chunk_max_words,
        overlap=settings.chunk_overlap_words,
    )


def make_embedder(settings: RetrievalSettings, *, allow_download: bool = False) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(
        settings.embedding_model,
        settings.embedding_revision,
        query_prefix=settings.query_prefix,
        device=settings.device,
        batch_size=settings.embedding_batch_size,
        allow_download=allow_download,
    )


def build_retriever(
    settings: RetrievalSettings | None = None,
    *,
    store: CorpusStore | None = None,
    embedder: Embedder | None = None,
    qdrant_client: Any | None = None,
) -> RetrievalBundle:
    """Never raises for missing artifacts: failures are reported in ``statuses`` and ``retriever`` is None
    when the required retriever cannot be built. Hybrid without dense degrades to lexical fallback."""
    s = settings or RetrievalSettings.from_env()
    statuses: list[ComponentStatus] = []
    try:
        store = store or load_store(s)
        statuses.append(ComponentStatus(name="corpus", ok=True, required=True, version=store.corpus_version))
    except ArtifactUnavailableError as exc:
        statuses.append(ComponentStatus(name="corpus", ok=False, required=True, detail=str(exc)))
        return RetrievalBundle(None, None, statuses)
    except Exception as exc:  # corrupt corpus etc.
        statuses.append(ComponentStatus(name="corpus", ok=False, required=True, detail=type(exc).__name__))
        return RetrievalBundle(None, None, statuses)

    # dense_fallback: dense ranking, BM25 only as a fallback; neither is individually required.
    fallback_mode = s.retriever == "dense_fallback"
    lexical: BM25Retriever | None = None
    if s.retriever in ("bm25", "hybrid", "dense_fallback"):
        try:
            lexical = BM25Retriever.from_settings(store, s)
            statuses.append(
                ComponentStatus(
                    name="retriever:bm25", ok=True, required=not fallback_mode, version=lexical.index_version
                )
            )
        except ArtifactUnavailableError as exc:
            statuses.append(
                ComponentStatus(
                    name="retriever:bm25", ok=False, required=not fallback_mode, degraded=fallback_mode, detail=str(exc)
                )
            )

    dense: DenseRetriever | None = None
    dense_error: str | None = None
    if s.retriever in ("dense", "hybrid", "dense_fallback"):
        try:
            # Check the active manifest first so a missing index never loads (or fetches) the model.
            active = resolve_active(s.index_dir, f"dense:{s.mode_suffix}")
            if s.embedding_revision is None:
                s = s.model_copy(update={"embedding_revision": active.definition.get("embedding_revision")})
            emb = embedder or make_embedder(s)
            client = qdrant_client if qdrant_client is not None else open_qdrant(s)
            dense = DenseRetriever.from_settings(store, s, emb, client)
            statuses.append(
                ComponentStatus(
                    name="retriever:dense", ok=True, required=s.retriever == "dense", version=dense.index_version
                )
            )
        except (ArtifactUnavailableError, ImportError) as exc:
            dense_error = str(exc)
            statuses.append(
                ComponentStatus(
                    name="retriever:dense",
                    ok=False,
                    required=s.retriever == "dense",
                    degraded=s.retriever in ("hybrid", "dense_fallback"),
                    detail=dense_error,
                )
            )

    retriever: Retriever | None = None
    if s.retriever == "bm25":
        retriever = lexical
    elif s.retriever == "dense":
        retriever = dense
    elif fallback_mode and lexical is None:
        retriever = dense  # no fallback available; dense alone (None if it failed too)
        if dense is None:
            statuses.append(ComponentStatus(name="retriever", ok=False, required=True, detail="no retriever"))
    elif lexical is not None:
        reranker = None
        if s.reranker_model and dense is not None:
            try:
                reranker = CrossEncoderReranker(
                    s.reranker_model, s.reranker_revision, device=s.device, depth=s.rerank_depth
                )
            except Exception as exc:
                log.warning("reranker unavailable: %s", type(exc).__name__)
        retriever = HybridRetriever(
            store,
            lexical,
            dense,
            mode=s.index_text_mode,
            rrf_k=s.rrf_k,
            candidate_k=s.candidate_k,
            reranker=reranker,
            dense_error=dense_error,
            fuse=not fallback_mode,
        )
    return RetrievalBundle(retriever, store, statuses)


def require_retriever(bundle: RetrievalBundle) -> Retriever:
    if bundle.retriever is None:
        failed = [f"{st.name}: {st.detail}" for st in bundle.statuses if not st.ok]
        raise RetrieverUnavailableError("; ".join(failed) or "retriever unavailable")
    return bundle.retriever
