"""Download the pinned generator snapshot (safetensors only) and write a verified manifest.

Usage: .venv/bin/python scripts/training/download_model.py [--config configs/models/generator.yaml]

Each downloaded LFS file's sha256 is checked against the Hub's metadata for the pinned revision; the licence
declared on the model card and the LICENSE file hash are recorded. Output:
artifacts/models/manifests/base_model.json (tracked; contains no weights).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import huggingface_hub
from huggingface_hub import HfApi, snapshot_download

from medquad_qa.models.settings import DEFAULT_GENERATOR_CONFIG_PATH, GeneratorConfig

ALLOW = ["*.json", "*.safetensors", "merges.txt", "vocab.json", "LICENSE", "README.md"]
EXPECTED_LICENSE = "apache-2.0"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_GENERATOR_CONFIG_PATH))
    ap.add_argument("--out", default="artifacts/models/manifests/base_model.json")
    args = ap.parse_args()
    cfg = GeneratorConfig.from_yaml(args.config)

    info = HfApi().model_info(cfg.model_id, revision=cfg.revision, files_metadata=True)
    if info.sha != cfg.revision:
        print(f"revision mismatch: hub={info.sha} pinned={cfg.revision}", file=sys.stderr)
        return 2
    license_id = (info.card_data or {}).get("license") if info.card_data else None
    if license_id != EXPECTED_LICENSE:
        print(f"licence {license_id!r} != {EXPECTED_LICENSE!r}; refusing", file=sys.stderr)
        return 2

    snapshot = Path(snapshot_download(cfg.model_id, revision=cfg.revision, allow_patterns=ALLOW))
    hub_files = {s.rfilename: s for s in info.siblings or []}
    files = []
    for path in sorted(p for p in snapshot.iterdir() if p.is_file()):
        meta = hub_files.get(path.name)
        digest = sha256_file(path)
        lfs_sha = meta.lfs.sha256 if meta is not None and meta.lfs is not None else None
        if lfs_sha is not None and lfs_sha != digest:
            print(f"sha256 mismatch for {path.name}", file=sys.stderr)
            return 3
        files.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": digest,
                "lfs_sha256_verified": lfs_sha is not None,
            }
        )

    manifest = {
        "model_id": cfg.model_id,
        "revision": cfg.revision,
        "hub_sha": info.sha,
        "license": license_id,
        "gated": info.gated,
        "allow_patterns": ALLOW,
        "files": files,
        "total_bytes": sum(f["bytes"] for f in files),
        "snapshot_dir": str(snapshot),
        "huggingface_hub_version": huggingface_hub.__version__,
        "verified_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(files)} files, {manifest['total_bytes'] / 1e9:.2f} GB, licence {license_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
