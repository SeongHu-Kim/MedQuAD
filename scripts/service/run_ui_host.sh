#!/usr/bin/env bash
# Run the Streamlit demo as a host process against the API (default http://127.0.0.1:8000). 127.0.0.1 only.
# Owner: service-platform-engineer.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MEDQUAD_API_URL="${MEDQUAD_API_URL:-http://127.0.0.1:8000}"
exec "$ROOT/.venv/bin/streamlit" run "$ROOT/src/medquad_qa/ui/app.py" \
  --server.address 127.0.0.1 --server.port "${MEDQUAD_UI_PORT:-8501}" --server.headless true \
  --browser.gatherUsageStats false
