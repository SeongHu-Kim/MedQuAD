#!/usr/bin/env bash
# E4b TEST run (pre-declaration b502e60 runs table + amendment 1), batch-1, run alone (D-045), v2c adapter only.
# Resumable by example_id: if interrupted, rerun the same command (amendment 1 §F(b)); a partial run is never reported.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/evaluation/e4b_env.sh
OUT=artifacts/evaluation/e4b/runs
TIERS="${TIERS:-1 2 3}"
for t in $TIERS; do
  case $t in
    1) $PY --evalset $ES/test_track_a.jsonl --modes finetuned finetuned_rag --out-dir $OUT $EXPECT ;;
    2) $PY --evalset $ES/test_track_c.jsonl --modes finetuned_rag --out-dir $OUT $EXPECT
       $PY --evalset $ES/test_track_c.jsonl --modes finetuned --notes-prefix personalized_advice general_control --out-dir $OUT $EXPECT ;;
    3) $PY --evalset $ES/train_probe.jsonl --modes finetuned --out-dir $OUT $EXPECT ;;
  esac
  echo "TIER_${t}_DONE $(date -u +%FT%TZ)"
done
