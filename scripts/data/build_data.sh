#!/usr/bin/env bash
# Rebuild and verify all data artifacts, re-render the audit report, run data tests.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}

"$PY" -m medquad_qa.data build
"$PY" -m medquad_qa.data verify
"$PY" -m medquad_qa.data report
"$PY" -m pytest tests/data -q
