"""End-to-end pipeline + CLI tests on a SYNTHETIC CSV; one opt-in test on the real file (real_data)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from data_synthetic import BOILER, LONG_A, LONG_B, SYNTHETIC_ROWS, write_csv

from medquad_qa.contracts import ContractViolationError, MedicalRecord
from medquad_qa.data.__main__ import main
from medquad_qa.data.config import REPO_ROOT, BuildConfig, load_config
from medquad_qa.data.pipeline import run_build
from medquad_qa.data.question_types import load_rules


def _build(root: Path, cfg: BuildConfig):  # noqa: ANN202
    return run_build(root / cfg.source.path, cfg, load_rules())


def test_reconciliation_and_exclusions(synthetic_root: Path, synthetic_cfg: BuildConfig) -> None:
    res = _build(synthetic_root, synthetic_cfg)
    reasons = Counter(e.reason for e in res.exclusions)
    assert reasons == {"empty_answer": 1, "exact_duplicate_row": 1, "non_informative_answer": 2}
    assert len(res.records) + len(res.exclusions) == len(SYNTHETIC_ROWS)
    ids = [r.record_id for r in res.records]
    assert len(ids) == len(set(ids))
    dup = next(e for e in res.exclusions if e.reason == "exact_duplicate_row")
    canon = next(r for r in res.records if r.provenance.row_index == dup.canonical_row_index)
    assert canon.provenance.duplicate_row_indices == [dup.row_index]
    assert "collapsed_exact_duplicates" in canon.quality_flags
    short = next(r for r in res.records if r.answer == "Zetaosis is rare.")
    assert "short_answer" in short.quality_flags  # valid short answer kept, not treated as missing


def test_records_valid_and_raw_preserved(synthetic_root: Path, synthetic_cfg: BuildConfig) -> None:
    res = _build(synthetic_root, synthetic_cfg)
    for line in res.files["data/processed/corpus.jsonl"].decode().splitlines():
        MedicalRecord.model_validate_json(line)
    r = next(r for r in res.records if "Rest" in r.answer)
    assert r.answer_raw == "Rest  and   fluids are advised; do NOT use 5 mg doses."
    assert r.answer == "Rest and fluids are advised; do NOT use 5 mg doses."
    assert r.question == "What are the treatments for Alphaitis?" and r.question_type == "treatment"
    kappa = next(r for r in res.records if r.question.startswith("Do you have"))
    assert kappa.topic is None and "missing_topic" in kappa.quality_flags
    assert all(r.source_url is None and r.source_document_id is None for r in res.records)
    assert sum("boilerplate_answer" in r.quality_flags for r in res.records) == 3


def test_groups_and_leakage(synthetic_root: Path, synthetic_cfg: BuildConfig) -> None:
    res = _build(synthetic_root, synthetic_cfg)
    assert res.leakage["passed"], res.leakage["blocking_checks"]
    split_of_group: dict[str, set[str]] = {}
    for r, s in zip(res.records, res.splits, strict=True):
        split_of_group.setdefault(r.split_group_id, set()).add(s)
    assert all(len(v) == 1 for v in split_of_group.values())
    alpha = {r.split_group_id for r in res.records if r.topic and r.topic.lower() == "alphaitis"}
    assert len(alpha) == 1  # topic merged across sources and case
    assert set(res.splits) == {"train", "validation", "test"}


def test_byte_identical_rebuild(synthetic_root: Path, synthetic_cfg: BuildConfig) -> None:
    a = _build(synthetic_root, synthetic_cfg).files
    b = _build(synthetic_root, synthetic_cfg).files
    assert a.keys() == b.keys() and all(a[k] == b[k] for k in a)


def test_tracked_manifests_have_no_answer_text(synthetic_root: Path, synthetic_cfg: BuildConfig) -> None:
    res = _build(synthetic_root, synthetic_cfg)
    manifests = {k: v.decode() for k, v in res.files.items() if k.startswith("data/manifests/")}
    assert len(manifests) == 8
    for text in manifests.values():
        for snippet in (LONG_A[:40], LONG_B[:40], BOILER[:40], "Rest and fluids"):
            assert snippet not in text


def test_versions_in_manifests(synthetic_root: Path, synthetic_cfg: BuildConfig) -> None:
    res = _build(synthetic_root, synthetic_cfg)
    assert res.corpus_version.startswith("medquad-1.0.0-") and len(res.corpus_version) == len("medquad-1.0.0-") + 12
    assert res.split_version.startswith(f"split-{synthetic_cfg.split.seed}-")
    exports = json.loads(res.files["data/manifests/exports_manifest.json"])
    assert sum(e["records"] for e in exports["splits"].values()) == len(res.records)
    assert exports["corpus_version"] == res.corpus_version


def test_bad_header_and_sha(synthetic_root: Path, synthetic_cfg: BuildConfig) -> None:
    write_csv(synthetic_root / "synthetic.csv", SYNTHETIC_ROWS[:3], header=["q", "a", "s", "f"])
    with pytest.raises(ContractViolationError):
        _build(synthetic_root, synthetic_cfg)
    write_csv(synthetic_root / "synthetic.csv", SYNTHETIC_ROWS)
    cfg = synthetic_cfg.model_copy(
        update={"source": synthetic_cfg.source.model_copy(update={"expected_sha256": "0" * 64})}
    )
    with pytest.raises(ContractViolationError):
        _build(synthetic_root, cfg)


def test_cli_build_verify_detects_drift(synthetic_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = ["--root", str(synthetic_root), "--config", str(synthetic_root / "build.yaml")]
    assert main(["build", *args]) == 0
    assert main(["verify", *args]) == 0
    assert "VERIFY OK" in capsys.readouterr().out
    target = synthetic_root / "data/manifests/split_manifest.jsonl"
    target.write_bytes(target.read_bytes() + b"\n")
    assert main(["verify", *args]) == 1
    (synthetic_root / "docs/data").mkdir(parents=True)
    assert main(["report", *args]) == 0
    assert "PASSED" in (synthetic_root / "docs/data/audit_report.md").read_text()


@pytest.mark.real_data
@pytest.mark.slow
def test_real_dataset_acceptance() -> None:
    cfg = load_config()
    src = REPO_ROOT / cfg.source.path
    if not src.exists():
        pytest.skip("medquad.csv not present")
    res = run_build(src, cfg, load_rules())
    assert len(res.records) + len(res.exclusions) == 16412
    assert len({r.record_id for r in res.records}) == len(res.records)
    assert res.leakage["passed"], res.leakage["blocking_checks"]
