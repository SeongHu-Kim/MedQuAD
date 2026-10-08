# Remediation fresh-retest environment (f4bf7e9, D-059..D-062). Sourced by remediation_retest.sh.
# The __SET_AT_FREEZE__ values MUST be replaced with the frozen remediation's runtime versions (read from the frozen
# commit) before the run; until then every stage fails its version guard by design.
export HF_HUB_OFFLINE=1 TORCH_DISABLE_NATIVE_JIT=1
export MEDQUAD_RETRIEVER=dense_fallback MEDQUAD_INDEX_TEXT_MODE=question_answer MEDQUAD_GATE_MODE=auto
unset MEDQUAD_QDRANT_URL MEDQUAD_ADAPTER_DIR || true
EXPECT_COMMON="--expect retriever=dense:qa --expect index_version=dense-qa-dc6b6a345fca --expect answerability_threshold_version=answerability-lexlr@3defcd00a31f:maxf1-val:0.307172 --expect model_version=Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554 --expect safety_rules_version=safety-v4+9be5f4d1 --expect prompt_version=rag-v1+df593554 --expect closed_book_prompt_version=cb-v1+a1f08aaf --expect evidence_filter_version=ef-v1+e899826d --expect safety_check_model=Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554"
EXPECT_V1="$EXPECT_COMMON --expect finetuned_model_version=Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554+lora:sft-main-20261006-081347-3f16e30e"
EXPECT_V2C="$EXPECT_COMMON --expect finetuned_model_version=Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554+lora:sft-main-20261007-104043-4c946221"
EXPECT_COMMIT="--expect-commit __SET_AT_FREEZE__"
PY=".venv/bin/python scripts/evaluation/run_fresh_retest.py"
ES_DIR=artifacts/evaluation/evalsets
