#!/usr/bin/env bash
# E4 Step 0 (D-048): DEV 15 items x 4 modes on the real frozen models; measures s/item before the TEST run.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/evaluation/e4_env.sh
$PY --evalset $ES/dev.jsonl --modes base rag finetuned finetuned_rag --limit 15 --out-dir artifacts/evaluation/e4/step0_dev $EXPECT
echo STEP0_DONE
