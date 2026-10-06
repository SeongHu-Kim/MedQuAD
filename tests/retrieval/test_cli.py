from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from medquad_qa.retrieval.cli import RUN_SCHEMA, main
from medquad_qa.retrieval.settings import RetrievalSettings


def _args(settings: RetrievalSettings) -> list[str]:
    return [
        "--corpus",
        str(settings.corpus_path),
        "--corpus-manifest",
        str(settings.corpus_manifest_path),
        "--index-dir",
        str(settings.index_dir),
        "--qdrant-path",
        str(settings.qdrant_path),
    ]


def test_cli_lifecycle_bm25(settings: RetrievalSettings, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a = _args(settings)
    assert main(["build", "--kind", "bm25", *a]) == 0
    built = json.loads(capsys.readouterr().out)
    assert set(built["active"]) == {"bm25:answer", "bm25:qa"}
    manifests = list((settings.index_dir / "manifests").glob("bm25-*.json"))
    assert len(manifests) == 2

    assert main(["verify", "--kind", "bm25", *a]) == 0
    assert all(v["ok"] for v in json.loads(capsys.readouterr().out).values())

    assert main(["list", *a]) == 0
    listed = json.loads(capsys.readouterr().out)["indexes"]
    assert len(listed) == 2 and all(r["active"] for r in listed)

    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        "\n".join(
            json.dumps({"example_id": f"q{i}", "question": q})
            for i, q in enumerate(["glimmer fever rash", "nothingmatcheszzz"])
        )
        + "\n"
    )
    out = tmp_path / "run.jsonl"
    assert main(["run", "--retriever", "bm25", "--top-k", "3", "--queries", str(queries), "--out", str(out), *a]) == 0
    capsys.readouterr()
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["example_id"] for r in rows] == ["q0", "q1"]
    assert rows[0]["retriever"] == "bm25:answer" and rows[0]["error"] is None
    assert set(rows[0]) == {
        "example_id",
        "retriever",
        "index_version",
        "corpus_version",
        "top_k",
        "latency_ms",
        "warnings",
        "error",
        "hits",
    }
    meta = json.loads((tmp_path / "run.jsonl.meta.json").read_text())
    assert meta["schema"] == RUN_SCHEMA and meta["n_queries"] == 2 and meta["collapse_duplicate_groups"] is False
    assert len(meta["queries_sha256"]) == 64 and "bm25:answer" in meta["config"]["indexes"]
    assert [h["rank"] for h in rows[0]["hits"]] == [1, 2, 3]
    assert set(rows[0]["hits"][0]) == {"record_id", "chunk_id", "rank", "score"}  # IDs only, no text
    assert rows[1]["hits"] == []
    from medquad_qa.evaluation.run_format import load_run  # evaluator-owned reader (D-024)

    assert [r.example_id for r in load_run(out, expected_example_ids=["q0", "q1"])] == ["q0", "q1"]

    assert main(["query", "--retriever", "bm25", "--mode", "question_answer", "--top-k", "2", *a, "glimmer"]) == 0
    assert json.loads(capsys.readouterr().out)["retriever"] == "bm25:qa"

    assert main(["delete-local", "--kind", "bm25", "--mode", "answer", *a]) == 0
    capsys.readouterr()
    assert main(["verify", "--kind", "bm25", *a]) == 0  # only bm25:qa remains active
    assert "bm25:answer" not in json.loads(capsys.readouterr().out)
    assert main(["query", "--retriever", "bm25", *a, "glimmer"]) == 2  # unavailable -> exit 2

    assert main(["rebuild", "--kind", "bm25", "--mode", "answer", *a]) == 0
    capsys.readouterr()
    assert main(["verify", *a]) == 0


def test_cli_verify_detects_corpus_drift(settings: RetrievalSettings, capsys: pytest.CaptureFixture[str]) -> None:
    a = _args(settings)
    assert main(["build", "--kind", "bm25", "--mode", "answer", *a]) == 0
    with settings.corpus_path.open("rb+") as fh:
        lines = fh.read().splitlines(keepends=True)
    settings.corpus_path.write_bytes(b"".join(lines[1:]))  # drop a record
    capsys.readouterr()
    assert main(["verify", *a]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["bm25:answer"]["problems"]


def test_lexical_import_path_has_no_torch_or_qdrant() -> None:
    code = (
        "import sys, medquad_qa.retrieval, medquad_qa.retrieval.factory, medquad_qa.retrieval.cli;"
        "print('torch' in sys.modules, 'qdrant_client' in sys.modules, 'sentence_transformers' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)  # noqa: S603
    assert out.stdout.strip() == "False False False"
