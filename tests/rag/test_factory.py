from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

from rag_fakes import FakeGenerator

from medquad_qa.contracts import ArtifactUnavailableError, QARequest
from medquad_qa.rag.factory import build_pipeline
from medquad_qa.rag.settings import RagSettings
from medquad_qa.retrieval.bm25 import build_bm25
from medquad_qa.retrieval.factory import load_store
from medquad_qa.retrieval.manifest import set_active, write_manifest
from medquad_qa.retrieval.settings import RetrievalSettings
from medquad_qa.retrieval.testing import write_synthetic_corpus


def _settings(tmp_path: Path, retriever: str = "bm25") -> RagSettings:
    corpus, manifest = write_synthetic_corpus(tmp_path / "data")
    rs = RetrievalSettings(
        retriever=retriever,
        corpus_path=corpus,
        corpus_manifest_path=manifest,  # type: ignore[arg-type]
        index_dir=tmp_path / "idx",
        qdrant_path=tmp_path / "q",
    )
    return RagSettings(retrieval=rs, gate_heuristic_path=tmp_path / "gate.json", gate_mode="heuristic")


def _provider(variant: str) -> FakeGenerator:
    if variant == "finetuned":
        raise ArtifactUnavailableError("no adapter")
    return FakeGenerator("Rest and fluids [E1].")


def test_build_pipeline_end_to_end_bm25(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    store = load_store(s.retrieval)
    m = build_bm25(store, s.retrieval)
    write_manifest(s.retrieval.index_dir, m)
    set_active(s.retrieval.index_dir, m.name, m.index_version)
    (tmp_path / "gate.json").write_text(json.dumps({"threshold": 0.5, "threshold_version": "heuristic-test"}))

    p = build_pipeline(s, generator_provider=_provider)
    st = {x.name: x for x in p.readiness()}
    assert st["corpus"].ok and st["retriever:bm25"].ok and st["generator:base"].ok
    assert not st["generator:finetuned"].ok and st["answerability"].version == "heuristic-test"
    assert p.available_modes() == frozenset({"base", "rag"})
    r = p.answer(QARequest(question="How is Glimmer fever treated?", experiment_mode="rag"), "x")
    assert not r.abstained and r.index_version == m.index_version and r.retriever == "bm25:answer"


def test_build_pipeline_missing_index_never_raises(tmp_path: Path) -> None:
    p = build_pipeline(_settings(tmp_path), generator_provider=_provider)
    st = {x.name: x for x in p.readiness()}
    assert not st["retriever:bm25"].ok
    assert p.available_modes() == frozenset({"base"})


def test_build_pipeline_without_model_factory_reports_unavailable(tmp_path: Path, monkeypatch) -> None:
    # Stub the models package so no real weights can ever load in an offline test.
    monkeypatch.setitem(sys.modules, "medquad_qa.models", types.ModuleType("medquad_qa.models"))
    p = build_pipeline(_settings(tmp_path))
    st = {x.name: x for x in p.readiness()}
    assert not st["generator:base"].ok and st["generator:base"].required
    assert p.available_modes() == frozenset()


def test_pipeline_import_graph_has_no_tensorflow(tmp_path: Path) -> None:  # D-036
    code = (
        "import sys; from medquad_qa.rag.factory import build_pipeline; from medquad_qa.rag.factory import "
        "_load_predictor; _load_predictor(); print('tensorflow' in sys.modules, 'keras' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)  # noqa: S603
    assert out.stdout.strip().splitlines()[-1] == "False False"
