#!/usr/bin/env bash
# Image-level rollback for the API service. Owner: service-platform-engineer.
#   snapshot  : tag the current medquad-api:<variant> as medquad-api:<variant>-prev (run BEFORE a rebuild)
#   rollback  : restart api-<variant> from medquad-api:<variant>-prev (no build) and wait for readiness
#   restore   : go back to medquad-api:<variant> (the newest build)
#   status    : show image IDs and the running container's image
# Artifacts (models/indexes/corpus) are mounted read-only and versioned by their manifests; rolling back an
# artifact means pointing the mounts/env at the previous version (docs/operations/runbook.md).
# Usage: VARIANT=gpu|cpu scripts/ops/rollback.sh <snapshot|rollback|restore|status>
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VARIANT="${VARIANT:-gpu}"
SERVICE="api-$VARIANT"
API="http://127.0.0.1:${MEDQUAD_API_PORT:-8000}"
COMPOSE=(docker compose -f "$ROOT/deploy/compose.yaml" --profile "$VARIANT")

wait_ready() {
  local t=0 code=000
  while (( t < ${READY_TIMEOUT:-900} )); do
    code=$(curl -s -o /dev/null -w '%{http_code}' "$API/health/ready" || true)
    [[ "$code" == "200" ]] && { echo "ready after ${t}s"; return 0; }
    sleep 5; t=$((t + 5))
  done
  echo "not ready after ${t}s (last $code)"; return 1
}

case "${1:-status}" in
  snapshot)
    docker image inspect "medquad-api:$VARIANT" >/dev/null
    docker tag "medquad-api:$VARIANT" "medquad-api:$VARIANT-prev"
    echo "tagged medquad-api:$VARIANT-prev = $(docker image inspect -f '{{.Id}}' "medquad-api:$VARIANT-prev")" ;;
  rollback)
    MEDQUAD_API_TAG="$VARIANT-prev" "${COMPOSE[@]}" up -d --no-build --force-recreate "$SERVICE"
    wait_ready ;;
  restore)
    MEDQUAD_API_TAG="$VARIANT" "${COMPOSE[@]}" up -d --no-build --force-recreate "$SERVICE"
    wait_ready ;;
  status)
    for t in "$VARIANT" "$VARIANT-prev"; do
      echo "medquad-api:$t -> $(docker image inspect -f '{{.Id}} {{.Created}}' "medquad-api:$t" 2>/dev/null || echo missing)"
    done
    cid=$("${COMPOSE[@]}" ps -q "$SERVICE" || true)
    [[ -n "$cid" ]] && echo "running $SERVICE image: $(docker inspect -f '{{.Config.Image}} {{.Image}}' "$cid")" ;;
  *) echo "usage: $0 <snapshot|rollback|restore|status>" >&2; exit 2 ;;
esac
