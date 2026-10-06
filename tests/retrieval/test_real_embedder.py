"""Real BGE embedder on the synthetic corpus. Opt-in: ``pytest -m real_model`` (needs the cached snapshot)."""

from __future__ import annotations

import pytest

from medquad_qa.contracts import ArtifactUnavailableError
from medquad_qa.retrieval.dense import DenseRetriever, build_dense, open_qdrant
from medquad_qa.retrieval.factory import load_store, make_embedder
from medquad_qa.retrieval.manifest import set_active, write_manifest
from medquad_qa.retrieval.settings import DEFAULT_EMBEDDING_REVISION, RetrievalSettings

pytestmark = pytest.mark.real_model


def test_bge_dense_on_synthetic(settings: RetrievalSettings) -> None:
    try:
        emb = make_embedder(settings)  # never downloads
    except ArtifactUnavailableError:
        pytest.skip("BGE snapshot not cached; run `python -m medquad_qa.retrieval build --kind dense` first")
    assert emb.revision == DEFAULT_EMBEDDING_REVISION and emb.dim == 384
    store = load_store(settings)
    client = open_qdrant(settings)
    try:
        m = build_dense(store, settings, emb, client)
        write_manifest(settings.index_dir, m)
        set_active(settings.index_dir, m.name, m.index_version)
        r = DenseRetriever.from_settings(store, settings, emb, client)
        hits = r.retrieve("Which medicine is used for Glimmer fever?", 3)
        assert "zorbatine" in hits[0].evidence_text
    finally:
        client.close()
