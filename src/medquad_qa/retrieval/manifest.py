"""Content-addressed index versions and index manifests.

Layout under ``index_dir`` (default ``artifacts/indexes``):
  manifests/<index_version>.json   tracked in git (IDs, hashes, params; no corpus text)
  manifests/active.json            {"bm25:answer": "<index_version>", ...} -- what the retrievers load
  bm25/<index_version>/            bm25s arrays + chunk id list (gitignored)
  qdrant_local/                    embedded Qdrant storage (gitignored; collection = index_version)
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from medquad_qa.contracts import RetrieverUnavailableError
from medquad_qa.retrieval.corpus import sha256_file

MANIFEST_SCHEMA = "medquad-index-manifest-v1"
IndexKind = Literal["bm25", "dense"]


class IndexManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = MANIFEST_SCHEMA
    index_version: str
    kind: IndexKind
    index_text_mode: Literal["answer", "question_answer"]
    corpus_version: str
    corpus_sha256: str | None
    n_records: int
    n_chunks: int
    texts_sha256: str = Field(description="Order-sensitive hash of (chunk_id, indexed text).")
    definition: dict[str, Any] = Field(description="All index-defining parameters (hashed into index_version).")
    files: dict[str, str] = Field(default_factory=dict, description="Relative path -> sha256 (bm25 only).")
    qdrant: dict[str, Any] | None = None
    build_seconds: float | None = None
    created_at: str
    environment: dict[str, str] = Field(default_factory=dict)

    @property
    def name(self) -> str:
        suffix = "qa" if self.index_text_mode == "question_answer" else "answer"
        return f"{self.kind}:{suffix}"


def compute_index_version(
    kind: str, mode: str, definition: dict[str, Any], texts_sha256: str, corpus_version: str
) -> str:
    payload = json.dumps(
        {"kind": kind, "mode": mode, "definition": definition, "texts": texts_sha256, "corpus": corpus_version},
        sort_keys=True,
        separators=(",", ":"),
    )
    suffix = "qa" if mode == "question_answer" else "answer"
    return f"{kind}-{suffix}-{hashlib.sha256(payload.encode()).hexdigest()[:12]}"


def environment_info() -> dict[str, str]:
    info = {"python": platform.python_version(), "machine": platform.machine()}
    for pkg in ("bm25s", "qdrant_client", "sentence_transformers", "transformers", "torch", "Stemmer"):
        try:
            mod = __import__(pkg)
            info[pkg] = str(getattr(mod, "__version__", "unknown"))
        except ImportError:
            continue
    return info


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def manifests_dir(index_dir: Path) -> Path:
    return index_dir / "manifests"


def manifest_path(index_dir: Path, index_version: str) -> Path:
    return manifests_dir(index_dir) / f"{index_version}.json"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_manifest(index_dir: Path, manifest: IndexManifest) -> Path:
    path = manifest_path(index_dir, manifest.index_version)
    _atomic_write(path, manifest.model_dump_json(indent=2) + "\n")
    return path


def load_manifest(index_dir: Path, index_version: str) -> IndexManifest:
    path = manifest_path(index_dir, index_version)
    if not path.is_file():
        raise RetrieverUnavailableError(f"index manifest not found: {path}")
    return IndexManifest.model_validate_json(path.read_text(encoding="utf-8"))


def list_manifests(index_dir: Path) -> list[IndexManifest]:
    d = manifests_dir(index_dir)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.json")):
        if p.name == "active.json":
            continue
        try:
            out.append(IndexManifest.model_validate_json(p.read_text(encoding="utf-8")))
        except ValueError:
            continue
    return out


def read_active(index_dir: Path) -> dict[str, str]:
    path = manifests_dir(index_dir) / "active.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in data.items()}


def set_active(index_dir: Path, name: str, index_version: str | None) -> None:
    active = read_active(index_dir)
    if index_version is None:
        active.pop(name, None)
    else:
        active[name] = index_version
    _atomic_write(manifests_dir(index_dir) / "active.json", json.dumps(active, indent=2, sort_keys=True) + "\n")


def resolve_active(index_dir: Path, name: str) -> IndexManifest:
    """Load the active manifest for ``name`` (e.g. 'bm25:answer') or raise RetrieverUnavailableError."""
    version = read_active(index_dir).get(name)
    if version is None:
        raise RetrieverUnavailableError(f"no active index for '{name}' under {index_dir} (run the build command)")
    return load_manifest(index_dir, version)


def file_hashes(root: Path, base: Path) -> dict[str, str]:
    return {str(p.relative_to(base)): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}


def verify_files(manifest: IndexManifest, base: Path) -> list[str]:
    """Return a list of drift problems (missing/changed/extra files); empty means OK."""
    problems: list[str] = []
    for rel, digest in sorted(manifest.files.items()):
        p = base / rel
        if not p.is_file():
            problems.append(f"missing file: {rel}")
        elif sha256_file(p) != digest:
            problems.append(f"sha256 mismatch: {rel}")
    return problems
