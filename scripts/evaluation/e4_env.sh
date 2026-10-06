# E4 frozen-run environment and expected versions (D-037, D-042, D-043, D-048). Sourced by e4_step0.sh / e4_full.sh.
export HF_HUB_OFFLINE=1 TORCH_DISABLE_NATIVE_JIT=1
export MEDQUAD_RETRIEVER=dense_fallback MEDQUAD_INDEX_TEXT_MODE=question_answer MEDQUAD_GATE_MODE=auto
unset MEDQUAD_QDRANT_URL || true
EXPECT="--expect safety_rules_version=safety-v2+43836b6c --expect prompt_version=rag-v1+df593554 --expect closed_book_prompt_version=cb-v1+a1f08aaf --expect retriever=dense:qa --expect index_version=dense-qa-dc6b6a345fca --expect answerability_threshold_version=answerability-lexlr@3defcd00a31f:maxf1-val:0.307172 --expect model_version=Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554 --expect finetuned_model_version=Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554+lora:sft-main-20261006-081347-3f16e30e"
PY=".venv/bin/python scripts/evaluation/run_e4_pipeline.py"
ES=artifacts/evaluation/evalsets
