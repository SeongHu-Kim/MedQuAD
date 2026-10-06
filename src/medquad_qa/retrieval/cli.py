"""Index lifecycle CLI: ``python -m medquad_qa.retrieval build|verify|rebuild|delete-local|list|query|run``."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from medquad_qa.contracts import MedQuADError
from medquad_qa.retrieval.bm25 import build_bm25
from medquad_qa.retrieval.dense import build_dense, open_qdrant, verify_dense
from medquad_qa.retrieval.factory import build_retriever, load_store, make_embedder, require_retriever
from medquad_qa.retrieval.manifest import (
    IndexManifest,
    list_manifests,
    load_manifest,
    manifest_path,
    read_active,
    set_active,
    verify_files,
    write_manifest,
)
from medquad_qa.retrieval.settings import MODE_SUFFIX, RetrievalSettings
from medquad_qa.retrieval.store import CorpusStore

RUN_SCHEMA = "medquad-retrieval-run-v1"


def _modes(arg: str) -> list[str]:
    return ["answer", "question_answer"] if arg == "all" else [arg]


def _kinds(arg: str) -> list[str]:
    return ["bm25", "dense"] if arg == "all" else [arg]


def _settings(args: argparse.Namespace, **extra: Any) -> RetrievalSettings:
    overrides: dict[str, Any] = dict(extra)
    if getattr(args, "index_dir", None):
        overrides["index_dir"] = Path(args.index_dir)
    if getattr(args, "corpus", None):
        overrides["corpus_path"] = Path(args.corpus)
    if getattr(args, "corpus_manifest", None):
        overrides["corpus_manifest_path"] = Path(args.corpus_manifest)
    if getattr(args, "qdrant_path", None):
        overrides["qdrant_path"] = Path(args.qdrant_path)
    if getattr(args, "no_stem", False):
        overrides["use_stemmer"] = False
    if getattr(args, "reranker", None):
        overrides["reranker_model"] = args.reranker
    if getattr(args, "collapse_duplicates", False):
        overrides["collapse_duplicates"] = True
    return RetrievalSettings.from_env(**overrides)


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False))


def cmd_build(args: argparse.Namespace) -> int:
    base = _settings(args)
    store = load_store(base)
    results = []
    embedder = None
    client = None
    for kind in _kinds(args.kind):
        for mode in _modes(args.mode):
            s = base.model_copy(update={"index_text_mode": mode})
            if kind == "bm25":
                manifest = build_bm25(store, s)
            else:
                embedder = embedder or make_embedder(s, allow_download=True)
                client = client if client is not None else open_qdrant(s)
                manifest = build_dense(store, s, embedder, client)
            write_manifest(s.index_dir, manifest)
            if not args.no_activate:
                set_active(s.index_dir, manifest.name, manifest.index_version)
            results.append(_summary(manifest))
    _print({"built": results, "active": read_active(base.index_dir)})
    return 0


def _summary(m: IndexManifest) -> dict[str, Any]:
    return {
        "name": m.name,
        "index_version": m.index_version,
        "corpus_version": m.corpus_version,
        "n_records": m.n_records,
        "n_chunks": m.n_chunks,
        "build_seconds": m.build_seconds,
    }


def _verify_one(m: IndexManifest, s: RetrievalSettings, store: CorpusStore, client: Any) -> list[str]:
    from medquad_qa.retrieval.corpus import texts_fingerprint

    problems: list[str] = []
    if m.kind == "bm25":
        problems += verify_files(m, s.index_dir)
        if store.corpus_version != m.corpus_version:
            problems.append(f"corpus_version {store.corpus_version} != manifest {m.corpus_version}")
        if texts_fingerprint(store.index_texts(m.index_text_mode), store.chunk_ids()) != m.texts_sha256:
            problems.append("indexed texts drifted from the loaded corpus")
    else:
        problems += verify_dense(m, client, store)
    if m.corpus_sha256 and store.corpus_sha256 and m.corpus_sha256 != store.corpus_sha256:
        problems.append("corpus file sha256 differs from manifest")
    return problems


def cmd_verify(args: argparse.Namespace) -> int:
    s = _settings(args)
    store = load_store(s)
    active = read_active(s.index_dir)
    report: dict[str, Any] = {}
    client = None
    ok = True
    for name, version in sorted(active.items()):
        kind = name.split(":")[0]
        if args.kind != "all" and kind != args.kind:
            continue
        m = load_manifest(s.index_dir, version)
        if kind == "dense" and client is None:
            try:
                client = open_qdrant(s)
            except MedQuADError as exc:
                report[name] = {"index_version": version, "ok": False, "problems": [str(exc)]}
                ok = False
                continue
        problems = _verify_one(m, s, store, client)
        report[name] = {"index_version": version, "ok": not problems, "problems": problems}
        ok = ok and not problems
    if not report:
        report["_"] = {"ok": False, "problems": ["no active indexes"]}
        ok = False
    _print(report)
    return 0 if ok else 1


def _delete_version(s: RetrievalSettings, version: str, purge_manifest: bool) -> dict[str, Any]:
    m = load_manifest(s.index_dir, version)
    index_root = s.index_dir.resolve()
    removed: list[str] = []
    if m.kind == "bm25":
        target = (s.index_dir / "bm25" / version).resolve()
        if index_root not in target.parents:
            raise MedQuADError(f"refusing to delete outside {index_root}: {target}")
        if target.exists():
            shutil.rmtree(target)
            removed.append(str(target))
    else:
        client = open_qdrant(s)
        collection = (m.qdrant or {}).get("collection", version)
        if client.collection_exists(collection):
            client.delete_collection(collection)
            removed.append(f"qdrant:{collection}")
    if read_active(s.index_dir).get(m.name) == version:
        set_active(s.index_dir, m.name, None)
    if purge_manifest:
        manifest_path(s.index_dir, version).unlink(missing_ok=True)
        removed.append(f"manifest:{version}")
    return {"index_version": version, "removed": removed}


def cmd_delete_local(args: argparse.Namespace) -> int:
    s = _settings(args)
    versions: list[str] = list(args.index_version or [])
    if not versions:
        active = read_active(s.index_dir)
        for kind in _kinds(args.kind):
            for mode in _modes(args.mode):
                v = active.get(f"{kind}:{MODE_SUFFIX[mode]}")
                if v:
                    versions.append(v)
    _print({"deleted": [_delete_version(s, v, args.purge_manifest) for v in versions]})
    return 0


def cmd_rebuild(args: argparse.Namespace) -> int:
    s = _settings(args)
    active = read_active(s.index_dir)
    for kind in _kinds(args.kind):
        for mode in _modes(args.mode):
            v = active.get(f"{kind}:{MODE_SUFFIX[mode]}")
            if v:
                _delete_version(s, v, purge_manifest=False)
    args.no_activate = False
    return cmd_build(args)


def cmd_list(args: argparse.Namespace) -> int:
    s = _settings(args)
    active = read_active(s.index_dir)
    rows = []
    for m in list_manifests(s.index_dir):
        row = _summary(m)
        row["active"] = active.get(m.name) == m.index_version
        row["created_at"] = m.created_at
        rows.append(row)
    _print({"index_dir": str(s.index_dir), "indexes": rows})
    return 0


def _retriever_for(args: argparse.Namespace) -> Any:
    s = _settings(args, retriever=args.retriever, index_text_mode=args.mode)
    return require_retriever(build_retriever(s))


def _retrieve(retriever: Any, query: str, top_k: int) -> tuple[list[Any], str, list[str]]:
    if hasattr(retriever, "retrieve_with_info"):
        hits, name, warnings = retriever.retrieve_with_info(query, top_k)
        return hits, name, warnings
    return retriever.retrieve(query, top_k), retriever.name, []


def cmd_query(args: argparse.Namespace) -> int:
    retriever = _retriever_for(args)
    t0 = time.perf_counter()
    hits, name, warnings = _retrieve(retriever, args.text, args.top_k)
    ms = (time.perf_counter() - t0) * 1000
    _print(
        {
            "retriever": name,
            "warnings": warnings,
            "latency_ms": round(ms, 2),
            "hits": [
                {
                    "rank": h.rank,
                    "record_id": h.record_id,
                    "chunk_id": h.chunk_id,
                    "score": round(h.score, 6),
                    "topic": h.topic,
                    "source_name": h.source_name,
                    "evidence_preview": h.evidence_text[:160],
                }
                for h in hits
            ],
        }
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Per-query ranked hits for frozen evaluation queries (IDs and scores only; no evidence text)."""
    retriever = _retriever_for(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    total_ms = 0.0
    with Path(args.queries).open(encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            item = json.loads(line)
            t0 = time.perf_counter()
            hits, name, warnings = _retrieve(retriever, item["question"], args.top_k)
            ms = (time.perf_counter() - t0) * 1000
            total_ms += ms
            row = {
                "schema": RUN_SCHEMA,
                "example_id": item["example_id"],
                "retriever": name,
                "index_version": retriever.index_version,
                "corpus_version": retriever.corpus_version,
                "top_k": args.top_k,
                "latency_ms": round(ms, 3),
                "warnings": warnings,
                "hits": [
                    {"record_id": h.record_id, "chunk_id": h.chunk_id, "rank": h.rank, "score": h.score} for h in hits
                ],
            }
            fout.write(json.dumps(row, sort_keys=True) + "\n")
            n += 1
    _print(
        {
            "out": str(out),
            "queries": n,
            "retriever": retriever.name,
            "mean_latency_ms": round(total_ms / n, 3) if n else None,
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m medquad_qa.retrieval", description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--index-dir")
    common.add_argument("--corpus")
    common.add_argument("--corpus-manifest")
    common.add_argument("--qdrant-path", help="Embedded Qdrant storage path (local mode).")
    common.add_argument("--no-stem", action="store_true", help="Disable PyStemmer stemming for BM25.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def kind_mode(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--kind", choices=["bm25", "dense", "all"], default="all")
        sp.add_argument("--mode", choices=["answer", "question_answer", "all"], default="all")

    b = sub.add_parser("build", parents=[common], help="Build indexes and activate them.")
    kind_mode(b)
    b.add_argument("--no-activate", action="store_true")
    b.set_defaults(func=cmd_build)

    v = sub.add_parser("verify", parents=[common], help="Verify active indexes against manifests and corpus.")
    v.add_argument("--kind", choices=["bm25", "dense", "all"], default="all")
    v.set_defaults(func=cmd_verify)

    r = sub.add_parser("rebuild", parents=[common], help="Delete local index data, then build again.")
    kind_mode(r)
    r.set_defaults(func=cmd_rebuild)

    d = sub.add_parser("delete-local", parents=[common], help="Delete local index data (keeps manifests by default).")
    kind_mode(d)
    d.add_argument("--index-version", action="append")
    d.add_argument("--purge-manifest", action="store_true")
    d.set_defaults(func=cmd_delete_local)

    ls = sub.add_parser("list", parents=[common], help="List index manifests.")
    ls.set_defaults(func=cmd_list)

    for name, fn in (("query", cmd_query), ("run", cmd_run)):
        q = sub.add_parser(name, parents=[common])
        q.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], default="hybrid")
        q.add_argument("--mode", choices=["answer", "question_answer"], default="answer")
        q.add_argument("--top-k", type=int, default=10)
        q.add_argument("--reranker", help="Cross-encoder model id (hybrid only).")
        q.add_argument("--collapse-duplicates", action="store_true")
        if name == "query":
            q.add_argument("text")
        else:
            q.add_argument("--queries", required=True, help="JSONL with example_id and question.")
            q.add_argument("--out", required=True)
        q.set_defaults(func=fn)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except MedQuADError as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
