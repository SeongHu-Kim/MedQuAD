#!/usr/bin/env bash
set -uo pipefail
REPO=/home/hangds/Desktop/SeongHuKim/MedQuAD
COMMIT=619228d
cd "$REPO" && mkdir -p artifacts/service/logs || { echo "ERROR: cd/mkdir failed" >&2; exit 2; }
LOG="artifacts/service/logs/e4_build_${COMMIT}_$(date -u +%Y%m%dT%H%M%SZ).log"
if [ -e "$LOG" ]; then echo "ERROR: $LOG exists; refusing to overwrite" >&2; exit 2; fi
if pgrep -f 'build_from_commit\.sh|buildx build|docker build' >/dev/null; then echo "ERROR: a build is already running" >&2; exit 3; fi
start=$(date +%s)
VARIANT=gpu bash scripts/ops/build_from_commit.sh "$COMMIT" 2>&1 | tee "$LOG" | tail -n 40
st=("${PIPESTATUS[@]}")
elapsed=$(( $(date +%s) - start ))
echo "STATUS build_exit=${st[0]} tee_exit=${st[1]} tail_exit=${st[2]} elapsed_s=${elapsed} log=${LOG}"
if [ "${st[0]}" -ne 0 ]; then echo "ERROR: build failed; see ${LOG}" >&2; exit "${st[0]}"; fi
if [ "${st[1]}" -ne 0 ]; then echo "ERROR: logging (tee) failed" >&2; exit 4; fi
exit 0
