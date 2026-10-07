"""Static security checks over tracked source (T12 secrets, T17 unsafe deserialisation / remote code)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").rglob("*.py"))

UNSAFE = {
    "pickle.load": re.compile(r"\bpickle\.loads?\("),
    "joblib.load": re.compile(r"\bjoblib\.load\("),
    "torch.load without weights_only=True": re.compile(r"\btorch\.load\((?![^)]*weights_only\s*=\s*True)"),
    "trust_remote_code=True": re.compile(r"trust_remote_code\s*=\s*True"),
    "yaml.load without SafeLoader": re.compile(r"\byaml\.load\((?![^)]*SafeLoader)"),
    "eval/exec": re.compile(r"(?<![\w.])(?:eval|exec)\("),
    "shell=True": re.compile(r"shell\s*=\s*True"),
}


@pytest.mark.parametrize("name", sorted(UNSAFE))
def test_no_unsafe_calls(name: str) -> None:
    hits = [
        f"{p.relative_to(ROOT)}:{i}"
        for p in SRC
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if UNSAFE[name].search(line) and not line.lstrip().startswith("#")
    ]
    assert not hits, hits


# Reviewed false positives (path, detector, sha1 of the flagged literal as reported by detect-secrets).
REVIEWED_FALSE_POSITIVES = {
    (
        "tests/api/test_api.py",
        "Secret Keyword",
        "f580337b1cc1d9726ec955972c2e635dd90b7bb0",
    ),  # synthetic canary marker named SECRET, not a credential
    # D-053 evidence files (lead-owned, sha256-cited in D-053, must not be edited): Base64-entropy hits are
    # build-output path strings ("artifacts/models/builds/sft-mix-v2…"), reviewed 2026-10-07, not credentials.
    (
        "docs/evidence/D-053/e4b_step2_20261007T034907Z.log",
        "Base64 High Entropy String",
        "9a4591a9282d75d934b192fb365f4769582353f9",
    ),
    (
        "docs/evidence/D-053/e4b_step2_summary_20261007T034907Z.json",
        "Base64 High Entropy String",
        "9a4591a9282d75d934b192fb365f4769582353f9",
    ),
    (
        "docs/evidence/D-053/e4b_step3a_compare.log",
        "Base64 High Entropy String",
        "9a4591a9282d75d934b192fb365f4769582353f9",
    ),
    (
        "docs/evidence/D-053/e4b_step3b_20261007T082524Z.log",
        "Base64 High Entropy String",
        "6995af548d65bc8cdd5e6e7195795baaaa161f10",
    ),
    (
        "docs/evidence/D-053/e4b_v2c_20261007T093306Z.log",
        "Base64 High Entropy String",
        "da6ddbee9c17381d836b0fabc55b2e38565ef492",
    ),
}


def _tracked_files() -> list[str]:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git not available")
    out = subprocess.run([git, "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout  # noqa: S603
    return [f for f in out.splitlines() if (ROOT / f).is_file()]


def test_no_secrets_in_tracked_files() -> None:
    """Hex-entropy detector disabled: the repo is full of sha256 digests and git revisions (false positives)."""
    exe = shutil.which("detect-secrets", path=str(ROOT / ".venv/bin"))
    if exe is None:
        pytest.skip("detect-secrets not installed")
    files = [f for f in _tracked_files() if not f.endswith((".jsonl", ".lock"))]
    res = subprocess.run(  # noqa: S603
        [exe, "scan", "--disable-plugin", "HexHighEntropyString", *files],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    found = [
        (path, r["type"], r["hashed_secret"], r["line_number"])
        for path, rows in json.loads(res.stdout)["results"].items()
        for r in rows
        if (path, r["type"], r["hashed_secret"]) not in REVIEWED_FALSE_POSITIVES
    ]
    assert found == [], [(p, t, n) for p, t, _, n in found]


def test_env_file_not_tracked() -> None:
    tracked = set(_tracked_files())
    assert ".env" not in tracked and not any(f.endswith(".env") for f in tracked)
