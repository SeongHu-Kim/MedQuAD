"""CLI: python -m medquad_qa.data {build,audit,split,export,verify,report}.

build   - write every output (corpus, exclusions, manifests, splits, leakage report, exports)
audit   - write audit.json + source_dataset.json only
split   - write split manifest, its meta and the leakage report only
export  - write per-split exports and exports_manifest.json only
verify  - rebuild in memory; fail unless every output on disk is byte-identical and leakage checks pass
report  - render docs/data/audit_report.md from data/manifests/{audit,leakage_report,split_manifest.meta}.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from medquad_qa.data.config import DEFAULT_CONFIG, DEFAULT_QTYPE_CONFIG, REPO_ROOT, load_config
from medquad_qa.data.manifests import sha256_bytes, write_outputs
from medquad_qa.data.pipeline import run_build
from medquad_qa.data.question_types import load_rules


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m medquad_qa.data", description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["build", "audit", "split", "export", "verify", "report"])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--qtype-config", type=Path, default=DEFAULT_QTYPE_CONFIG)
    ap.add_argument("--root", type=Path, default=REPO_ROOT, help="Repo root that relative paths resolve against.")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    root: Path = args.root

    if args.command == "report":
        from medquad_qa.data.report import render_audit_report

        out = root / "docs" / "data" / "audit_report.md"
        out.write_text(render_audit_report(root / cfg.outputs.manifests_dir), encoding="utf-8")
        print(f"wrote {out.relative_to(root)}")
        return 0

    result = run_build(root / cfg.source.path, cfg, load_rules(args.qtype_config))
    leak_ok = bool(result.leakage["passed"])
    print(f"corpus_version={result.corpus_version} split_version={result.split_version}")
    print(f"records={len(result.records)} excluded={len(result.exclusions)} leakage_passed={leak_ok}")

    if args.command == "verify":
        mismatched = []
        for rel, data in sorted(result.files.items()):
            path = root / rel
            disk = path.read_bytes() if path.exists() else None
            same = disk is not None and sha256_bytes(disk) == sha256_bytes(data)
            print(f"{'OK  ' if same else 'DIFF'} {sha256_bytes(data)} {rel}")
            if not same:
                mismatched.append(rel)
        if mismatched or not leak_ok:
            print(f"VERIFY FAILED: {len(mismatched)} file(s) differ; leakage_passed={leak_ok}", file=sys.stderr)
            return 1
        print(f"VERIFY OK: {len(result.files)} files byte-identical; leakage checks passed")
        return 0

    selected = result.file_groups[args.command]
    write_outputs({rel: result.files[rel] for rel in selected}, root)
    for rel in selected:
        print(f"wrote {sha256_bytes(result.files[rel])} {rel}")
    if not leak_ok:
        print("LEAKAGE CHECKS FAILED; see leakage_report.json", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
