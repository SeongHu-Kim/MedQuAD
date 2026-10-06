#!/usr/bin/env bash
# The user's 6-step Docker acceptance sequence for the local stack. Owner: service-platform-engineer.
#   1) compose config  2) build + up  3) /health/live + /health/ready  4) real POST /v1/qa
#   5) /metrics + docker compose logs  6) summary (commands + results) for the lead/ledger
# Usage: PROFILE=gpu|cpu [MONITORING=1] [NO_BUILD=1] [READY_TIMEOUT=900] scripts/ops/stack_up_smoke.sh
# Evidence: artifacts/service/stack_smoke/<UTC timestamp>/ (gitignored except .md).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROFILE="${PROFILE:-gpu}"
READY_TIMEOUT="${READY_TIMEOUT:-900}"
API="http://127.0.0.1:${MEDQUAD_API_PORT:-8000}"
QUESTION="${STACK_SMOKE_QUESTION:-What are the symptoms of glaucoma?}"
MODE="${STACK_SMOKE_MODE:-rag}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$ROOT/artifacts/service/stack_smoke/$TS"
mkdir -p "$OUT"
COMPOSE=(docker compose -f "$ROOT/deploy/compose.yaml" --profile "$PROFILE")
[[ "${MONITORING:-0}" == "1" ]] && COMPOSE+=(--profile monitoring)
SUMMARY="$OUT/summary.md"
status=0

step() { echo; echo "=== $*"; echo "- $*" >> "$SUMMARY"; }
result() { echo "  - $*" | tee -a "$SUMMARY"; }
run() { echo "+ $*"; echo "  - \`$*\`" >> "$SUMMARY"; "$@"; }

{
  echo "# Stack smoke $TS"
  echo
  echo "profile=$PROFILE monitoring=${MONITORING:-0} git=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo none)"
  echo
} > "$SUMMARY"

step "1. compose config"
if run "${COMPOSE[@]}" config -q; then result "config valid"; else result "config INVALID"; exit 1; fi

step "2. build + up"
if [[ "${NO_BUILD:-0}" == "1" ]]; then up_args=(up -d --no-build); else up_args=(up -d --build); fi
if run "${COMPOSE[@]}" "${up_args[@]}" > "$OUT/up.log" 2>&1; then result "up ok"; else
  result "up FAILED (see up.log)"; tail -n 30 "$OUT/up.log"; exit 1; fi
"${COMPOSE[@]}" ps --format '{{.Service}} {{.State}} {{.Status}} {{.Ports}}' | tee "$OUT/ps.txt"

step "3. /health/live and /health/ready (timeout ${READY_TIMEOUT}s)"
live=000; ready=000; start=$(date +%s)
while (( $(date +%s) - start < READY_TIMEOUT )); do
  live=$(curl -s -o /dev/null -w '%{http_code}' "$API/health/live" || true)
  ready=$(curl -s -o "$OUT/ready.json" -w '%{http_code}' "$API/health/ready" || true)
  [[ "$ready" == "200" ]] && break
  if [[ "$live" == "200" ]] && grep -q '"pipeline":"failed"' "$OUT/ready.json" 2>/dev/null; then break; fi
  sleep 5
done
elapsed=$(( $(date +%s) - start ))
result "live=$live ready=$ready after ${elapsed}s; body: $(head -c 600 "$OUT/ready.json" 2>/dev/null)"
[[ "$live" == "200" && "$ready" == "200" ]] || status=1

step "4. real POST /v1/qa (mode=$MODE)"
payload=$(python3 -c 'import json,sys; print(json.dumps({"question": sys.argv[1], "experiment_mode": sys.argv[2], "top_k": 5}))' "$QUESTION" "$MODE")
qa_code=$(curl -s -o "$OUT/qa_response.json" -w '%{http_code}' -X POST -H 'content-type: application/json' \
  -H "X-Request-ID: stack-smoke-$TS" -d "$payload" --max-time 180 "$API/v1/qa" || true)
result "POST /v1/qa -> $qa_code"
python3 - "$OUT/qa_response.json" <<'EOF' | tee -a "$SUMMARY"
import json, sys
try:
    r = json.load(open(sys.argv[1]))
except Exception as exc:
    print(f"  - response not JSON ({type(exc).__name__})"); sys.exit()
keys = ["request_id", "experiment_mode", "abstained", "abstention_reason", "model_version", "corpus_version",
        "index_version", "prompt_version", "retriever", "latency_ms", "error_code"]
print("  - " + ", ".join(f"{k}={r.get(k)!r}" for k in keys if k in r))
if "citations" in r:
    print(f"  - citations={[c['record_id'] for c in r['citations']]} retrieved={r.get('retrieved_record_ids')}")
    print(f"  - invalid_citation_ids={r.get('invalid_citation_ids')} warnings={r.get('warnings')}")
EOF
[[ "$qa_code" == "200" ]] || status=1

step "5. /metrics and docker compose logs"
curl -s "$API/metrics" > "$OUT/metrics.txt" || true
grep -E '^medquad_(qa_requests_total|pipeline_state|artifact_info|component_ready|mode_available|qa_latency_seconds_count)' \
  "$OUT/metrics.txt" | tee "$OUT/metrics_excerpt.txt" | sed 's/^/  - /' >> "$SUMMARY"
cat "$OUT/metrics_excerpt.txt"
"${COMPOSE[@]}" logs --no-color --tail 200 > "$OUT/compose_logs.txt" 2>&1
result "compose logs saved: $(wc -l < "$OUT/compose_logs.txt") lines"
if grep -qF "$QUESTION" "$OUT/compose_logs.txt"; then
  result "PRIVACY FAIL: question text found in logs"; status=1
else
  result "privacy: question text not present in compose logs"
fi

step "6. summary"
result "overall: $([[ $status == 0 ]] && echo PASS || echo FAIL); evidence dir: ${OUT#"$ROOT"/}"
echo; cat "$SUMMARY"
exit "$status"
