# E4b (adapter v2c) frozen-run environment and expected versions. Sourced by e4b_step0.sh / e4b_full.sh.
# Identical to e4_env.sh except: MEDQUAD_ADAPTER_DIR selects the v2c adapter WITHOUT promotion (adapters/CURRENT stays
# v1), the expected finetuned_model_version is v2c, and outputs go to artifacts/evaluation/e4b/ (E4 tree untouched).
export HF_HUB_OFFLINE=1 TORCH_DISABLE_NATIVE_JIT=1
export MEDQUAD_RETRIEVER=dense_fallback MEDQUAD_INDEX_TEXT_MODE=question_answer MEDQUAD_GATE_MODE=auto
export MEDQUAD_ADAPTER_DIR=artifacts/models/adapters/sft-main-20261007-104043-4c946221
unset MEDQUAD_QDRANT_URL || true
EXPECT="--expect safety_rules_version=safety-v2+43836b6c --expect prompt_version=rag-v1+df593554 --expect closed_book_prompt_version=cb-v1+a1f08aaf --expect retriever=dense:qa --expect index_version=dense-qa-dc6b6a345fca --expect answerability_threshold_version=answerability-lexlr@3defcd00a31f:maxf1-val:0.307172 --expect model_version=Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554 --expect finetuned_model_version=Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554+lora:sft-main-20261007-104043-4c946221"
PY=".venv/bin/python scripts/evaluation/run_e4_pipeline.py"
ES=artifacts/evaluation/evalsets
