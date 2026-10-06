#!/usr/bin/env bash
# Run the API as a host process (D-017 option B, and for development). Binds 127.0.0.1 only.
# Qdrant/MLflow can still run in Compose: `docker compose -f deploy/compose.yaml up -d qdrant mlflow`.
# Owner: service-platform-engineer.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export MEDQUAD_BIND_HOST="${MEDQUAD_BIND_HOST:-127.0.0.1}"
export MEDQUAD_API_PORT="${MEDQUAD_API_PORT:-8000}"
export MEDQUAD_QDRANT_URL="${MEDQUAD_QDRANT_URL:-http://127.0.0.1:6333}"
export MEDQUAD_MLFLOW_TRACKING_URI="${MEDQUAD_MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
if [[ "$MEDQUAD_BIND_HOST" != "127.0.0.1" && "${MEDQUAD_ALLOW_NON_LOCAL_BIND:-0}" != "1" ]]; then
  echo "refusing to bind $MEDQUAD_BIND_HOST (set MEDQUAD_ALLOW_NON_LOCAL_BIND=1 to override)" >&2; exit 2
fi
exec .venv/bin/python -m medquad_qa.api
