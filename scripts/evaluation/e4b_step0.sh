#!/usr/bin/env bash
# E4b Step 0 (pre-declaration L14 / amendment 1): DEV 15 items x {finetuned, finetuned_rag} with the v2c adapter.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/evaluation/e4b_env.sh
$PY --evalset $ES/dev.jsonl --modes finetuned finetuned_rag --limit 15 --out-dir artifacts/evaluation/e4b/step0_dev $EXPECT
echo STEP0_DONE
