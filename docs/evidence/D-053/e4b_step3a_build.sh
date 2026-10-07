#!/usr/bin/env bash
# E4b step 3(a) (DRAFT, not yet approved to run): rebuild sft-mix-v2 with the r3 code (v2b keys absent:
# committed mix_v2.yaml + lora_sft_mixed_v2.yaml), CPU only, same environment as the original build, into a NEW
# directory under artifacts/models/builds/. Build only: the leakage gate is run separately by
# evaluation-safety-engineer (D-035), then e4b_step3a_compare.py compares against sft-mix-v2-20261007-035957.
# No training, no generator weights, no GPU, no staging/commit, no retries. Exit: 1 build, 2 out-dir, 5 artifacts/evaluation touched.
set -uo pipefail
REPO=/home/hangds/Desktop/SeongHuKim/MedQuAD
SP=/tmp/claude-1000/-home-hangds-Desktop-SeongHuKim-MedQuAD/8509dc6a-7edf-42b5-bc59-cbc01100fbfa/scratchpad
cd "$REPO" || exit 90
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$SP/e4b_step3a_$TS.log"
MARK="$SP/e4b_step3a_$TS.marker"
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

echo "start $TS  HEAD=$(git rev-parse HEAD)"
echo "code: sft_rag_data.py $(sha256sum src/medquad_qa/training/sft_rag_data.py | cut -c1-64)  train_lora.py $(sha256sum scripts/training/train_lora.py | cut -c1-64)"
echo "configs (committed, must be unmodified): $(git status --short -- configs/training/mix_v2.yaml configs/training/lora_sft_mixed_v2.yaml | wc -l) modified"

# stage 1: build (writes only artifacts/models/builds/sft-mix-v2-<UTC>/) - identical flags to the 2026-10-07 build
.venv/bin/python scripts/training/train_lora.py --mix v2 --build-only \
    --config configs/training/lora_sft_mixed_v2.yaml --mix-config configs/training/mix_v2.yaml \
    --prompt-builder pipeline --expect-prompt-version cb-v1+a1f08aaf \
    --expect-rag-prompt-version rag-v1+df593554
BUILD_EXIT=$?
echo "BUILD_EXIT=$BUILD_EXIT elapsed_s=$(( $(date +%s) - T0 ))"
[ "$BUILD_EXIT" -eq 0 ] || exit 1

# stage 2: locate the build dir (exactly one new sft-mix-v2-* dir created by this run)
mapfile -t DIRS < <(find artifacts/models/builds -mindepth 1 -maxdepth 1 -type d -name 'sft-mix-v2-*' -newer "$MARK")
[ "${#DIRS[@]}" -eq 1 ] || { echo "expected exactly 1 new build dir, found ${#DIRS[@]}: ${DIRS[*]:-}"; exit 2; }
echo "OUT=${DIRS[0]}"
find "${DIRS[0]}" -type f -print0 | sort -z | xargs -0 sha256sum
exit 0
