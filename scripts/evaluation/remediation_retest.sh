#!/usr/bin/env bash
# Fresh held-out remediation retest (f4bf7e9 §1/§2, D-059..D-062). Run ONCE, alone on the GPU, on the frozen commit.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/evaluation/remediation_env.sh
$PY --stage safety $EXPECT_COMMIT $EXPECT_V1
$PY --stage injection --modes rag finetuned_rag --label v1 $EXPECT_COMMIT $EXPECT_V1
MEDQUAD_ADAPTER_DIR=artifacts/models/adapters/sft-main-20261007-104043-4c946221 \
  $PY --stage injection --modes finetuned_rag --label v2c $EXPECT_COMMIT $EXPECT_V2C
echo "RETEST_GENERATION_DONE (finalize only after the §2.1 harm review is recorded)"
