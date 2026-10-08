#!/usr/bin/env bash
# f4bf7e9 §3 four-mode regression run vs E4 (same frozen sets, same tiers as e4_full.sh), on the frozen remediation
# commit, alone on the GPU. Generation only; then score with score_e4.py and compare with
# compare_remediation_regression.py. Resumable by example_id (rerun the same command).
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/evaluation/remediation_env.sh
COMMIT="${EXPECT_COMMIT#--expect-commit }"
.venv/bin/python - "$COMMIT" <<'PYEOF'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("rfr", "scripts/evaluation/run_fresh_retest.py")
rfr = importlib.util.module_from_spec(spec); spec.loader.exec_module(rfr)
from medquad_qa.evaluation.fresh_retest import GuardError, check_frozen_commit
try:
    check_frozen_commit(sys.argv[1], *rfr.frozen_state(sys.argv[1]))
except GuardError as exc:
    print(f"GUARD FAILED: {exc}", file=sys.stderr); sys.exit(4)
PYEOF
RUN=".venv/bin/python scripts/evaluation/run_e4_pipeline.py"
OUT=artifacts/evaluation/remediation/regression/runs
$RUN --evalset $ES_DIR/test_track_a.jsonl --modes base rag finetuned finetuned_rag --out-dir $OUT $EXPECT_V1
$RUN --evalset $ES_DIR/train_probe.jsonl --modes base finetuned --out-dir $OUT $EXPECT_V1
$RUN --evalset $ES_DIR/test_track_c.jsonl --modes rag finetuned_rag --out-dir $OUT $EXPECT_V1
$RUN --evalset $ES_DIR/test_track_c.jsonl --modes base finetuned --notes-prefix personalized_advice general_control --out-dir $OUT $EXPECT_V1
echo "REGRESSION_GENERATION_DONE $(date -u +%FT%TZ)"
