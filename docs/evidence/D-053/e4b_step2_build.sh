#!/usr/bin/env bash
# E4b step 2 (DRAFT, not yet approved to run): CPU-only v2 data build + leakage gate + read-only report.
# No training, no generator weights, no GPU, no staging/commit, no retries.
# Stops at the first failing stage. Exit: 1 build, 2 out-dir, 3 gate, 4 report, 5 artifacts/evaluation touched.
set -uo pipefail
REPO=/home/hangds/Desktop/SeongHuKim/MedQuAD
SP=/tmp/claude-1000/-home-hangds-Desktop-SeongHuKim-MedQuAD/8509dc6a-7edf-42b5-bc59-cbc01100fbfa/scratchpad
cd "$REPO" || exit 90
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$SP/e4b_step2_$TS.log"
MARK="$SP/e4b_step2_$TS.marker"
touch "$MARK"
export TORCH_DISABLE_NATIVE_JIT=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES= \
       MEDQUAD_MLFLOW_TRACKING_URI=off TOKENIZERS_PARALLELISM=false
exec > >(tee "$LOG") 2>&1
T0=$(date +%s)

final_checks() {  # always runs on exit: artifacts/evaluation must be untouched
    local rc=$?
    local touched
    touched=$(find artifacts/evaluation -newer "$MARK" -print)
    if [ -n "$touched" ]; then
        echo "ARTIFACTS_EVALUATION_TOUCHED:"; echo "$touched"
        [ "$rc" -eq 0 ] && rc=5
    else
        echo "artifacts/evaluation: nothing created or modified since $TS"
    fi
    echo "FINAL_EXIT=$rc total_elapsed_s=$(( $(date +%s) - T0 ))"
    exit "$rc"
}
trap final_checks EXIT

echo "start $TS  HEAD=$(git rev-parse HEAD)  env: CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' HF_HUB_OFFLINE=$HF_HUB_OFFLINE"

# stage 1: build (writes only artifacts/models/builds/sft-mix-v2-<UTC>/)
.venv/bin/python scripts/training/train_lora.py --mix v2 --build-only \
    --config configs/training/lora_sft_mixed_v2.yaml --mix-config configs/training/mix_v2.yaml \
    --prompt-builder pipeline --expect-prompt-version cb-v1+a1f08aaf \
    --expect-rag-prompt-version rag-v1+df593554 | tee "$SP/e4b_step2_summary_$TS.json"
BUILD_EXIT=${PIPESTATUS[0]}
echo "BUILD_EXIT=$BUILD_EXIT elapsed_s=$(( $(date +%s) - T0 ))"
[ "$BUILD_EXIT" -eq 0 ] || exit 1

# stage 2: locate the build dir (exactly one new sft-mix-v2-* dir created by this run)
mapfile -t DIRS < <(find artifacts/models/builds -mindepth 1 -maxdepth 1 -type d -name 'sft-mix-v2-*' -newer "$MARK")
[ "${#DIRS[@]}" -eq 1 ] || { echo "expected exactly 1 new build dir, found ${#DIRS[@]}: ${DIRS[*]:-}"; exit 2; }
OUT=${DIRS[0]}
echo "OUT=$OUT"

# stage 3: leakage gate (reads eval sets, split manifest, corpus, pairs manifests; writes only into $OUT)
.venv/bin/python scripts/evaluation/check_protected_leakage.py \
    --sft "$OUT/sft_record_ids.jsonl" \
    --pairs-train artifacts/models/answerability/pairs/pairs_manifest_train.jsonl \
    --pairs-validation artifacts/models/answerability/pairs/pairs_manifest_validation.jsonl \
    --threshold-ids "$OUT/val_record_ids.jsonl" \
    --out "$OUT/protected_check.json"
GATE_EXIT=$?
echo "GATE_EXIT=$GATE_EXIT"
[ "$GATE_EXIT" -eq 0 ] || exit 3

# stage 4: read-only report
.venv/bin/python -I "$SP/e4b_step2_report.py" "$OUT"
REPORT_EXIT=$?
echo "REPORT_EXIT=$REPORT_EXIT"
[ "$REPORT_EXIT" -eq 0 ] || exit 4

# hashes of every output file (printed to the log only; nothing written into $OUT)
find "$OUT" -type f -print0 | sort -z | xargs -0 sha256sum
exit 0
