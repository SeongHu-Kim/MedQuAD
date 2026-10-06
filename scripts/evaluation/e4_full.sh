#!/usr/bin/env bash
# E4 full TEST run (D-048), batch-1, run alone (D-045). Tiers in priority order; TIERS="1 2 3" to stop before tier 4.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/evaluation/e4_env.sh
OUT=artifacts/evaluation/e4/runs
TIERS="${TIERS:-1 2 3 4}"
for t in $TIERS; do
  case $t in
    1) $PY --evalset $ES/test_track_a.jsonl --modes base rag finetuned finetuned_rag --out-dir $OUT $EXPECT ;;
    2) $PY --evalset $ES/train_probe.jsonl --modes base finetuned --out-dir $OUT $EXPECT ;;
    3) $PY --evalset $ES/test_track_c.jsonl --modes rag finetuned_rag --out-dir $OUT $EXPECT ;;
    4) $PY --evalset $ES/test_track_c.jsonl --modes base finetuned --notes-prefix personalized_advice general_control --out-dir $OUT $EXPECT ;;
  esac
  echo "TIER_${t}_DONE $(date -u +%FT%TZ)"
done
