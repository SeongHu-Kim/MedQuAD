#!/usr/bin/env bash
# Smoke-test an API image WITHOUT artifacts: live must be 200, ready 503, POST /v1/qa 503 not_ready,
# /metrics must expose medquad_pipeline_state. Hardened run flags match deploy/compose.yaml.
# Usage: scripts/ops/image_smoke.sh <image>      Owner: service-platform-engineer.
set -euo pipefail
IMAGE="${1:?usage: image_smoke.sh <image>}"
PORT="${SMOKE_PORT:-18080}"
NAME="medquad-image-smoke-$$"
cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

docker run -d --name "$NAME" --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges \
  --user 10001:10001 -p "127.0.0.1:${PORT}:8000" "$IMAGE" >/dev/null

code() { curl -s -o /dev/null -w '%{http_code}' "$@" || true; }
for _ in $(seq 1 60); do
  [[ "$(code "http://127.0.0.1:${PORT}/health/live")" == "200" ]] && break
  sleep 1
done
fail=0
check() { local want="$1" got="$2" what="$3"; if [[ "$got" == "$want" ]]; then echo "ok   $what -> $got"; else echo "FAIL $what -> $got (want $want)"; fail=1; fi; }
check 200 "$(code "http://127.0.0.1:${PORT}/health/live")" "GET /health/live"
sleep 2  # let the background build fail (no rag artifacts in a bare image)
check 503 "$(code "http://127.0.0.1:${PORT}/health/ready")" "GET /health/ready"
check 503 "$(code -X POST -H 'content-type: application/json' -d '{"question":"What is synthetic X?"}' "http://127.0.0.1:${PORT}/v1/qa")" "POST /v1/qa"
check 422 "$(code -X POST -H 'content-type: application/json' -d '{"question":"x"}' "http://127.0.0.1:${PORT}/v1/qa")" "POST /v1/qa (invalid)"
if curl -s "http://127.0.0.1:${PORT}/metrics" | grep -q '^medquad_pipeline_state'; then echo "ok   /metrics exposes medquad_pipeline_state"; else echo "FAIL /metrics"; fail=1; fi
echo "user: $(docker exec "$NAME" id -u 2>/dev/null || echo '?')"
docker logs "$NAME" 2>&1 | tail -n 5
exit "$fail"
