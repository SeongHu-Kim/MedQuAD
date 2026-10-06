# Reproducing the data artifacts

Requirements: the project `.venv` (pydantic, PyYAML; no other dependencies) and `medquad.csv` at the repo
root. The file's sha256 must match `configs/data/build.yaml` → `source.expected_sha256`, or the build
stops.

```bash
# full build: corpus, exclusions, groups, splits, leakage report, exports, audit
.venv/bin/python -m medquad_qa.data build

# rebuild in memory and compare byte-for-byte with what is on disk; also re-checks leakage
.venv/bin/python -m medquad_qa.data verify

# re-render docs/data/audit_report.md from data/manifests/*.json
.venv/bin/python -m medquad_qa.data report

# partial writes (the computation is identical; only these outputs are written)
.venv/bin/python -m medquad_qa.data audit    # audit.json, source_dataset.json
.venv/bin/python -m medquad_qa.data split    # split_manifest.jsonl, its meta, leakage_report.json
.venv/bin/python -m medquad_qa.data export   # records_{train,validation,test}.jsonl, exports_manifest.json

# tests: offline, synthetic fixtures; test_real_dataset_acceptance runs only if medquad.csv exists
.venv/bin/python -m pytest tests/data -q
.venv/bin/python -m pytest tests/data -q -m "not real_data"   # strictly synthetic
```

`scripts/data/build_data.sh` runs build, verify, report and the tests in sequence.

## Exit codes

| command | 0 | non-zero |
|---|---|---|
| `build`/`audit`/`split`/`export` | outputs written, leakage passed | 2 = leakage checks failed (outputs still written for inspection) |
| `verify` | all files byte-identical and leakage passed | 1 = drift or leakage failure; each differing file is printed as `DIFF` |

## Determinism

- No timestamps, absolute paths or unordered iteration reach any output.
- JSON is written with sorted keys.
- The split shuffle uses a string-seeded `random.Random`, which is stable across processes.
- A rebuild under a different `PYTHONHASHSEED` produced sha256-identical files (evidence is reported
  to the lead).

## Changing parameters

Edit `configs/data/build.yaml` or `configs/data/question_types.yaml` (bump `rule_version`), then run
`build`, `report` and `verify`. Any change to corpus content changes `corpus_version`, and any change to
the split changes `split_version`. Retrieval indexes, training runs and evaluation sets built on the old
versions must be rebuilt.
