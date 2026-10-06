# Team Migration Handoff (Phase 1 → real agent team)

Owner: lead. Written 2026-10-06 (session 2). Purpose: let a new lead session launch a **real Claude Code agent team** and release implementation **without repeating discovery or planning**.

Status: **implementation PAUSED**. Phase 1 (discovery and planning) is COMPLETE. Gate G1 rulings are recorded in `docs/decisions.md` (D-009 to D-018). START IMPLEMENTATION has **not** been sent.

## 1. Why the migration is needed (execution-mode finding)

| Evidence | Value |
|---|---|
| Session interface | VS Code extension: `CLAUDE_CODE_ENTRYPOINT=claude-vscode`, native binary `~/.vscode/extensions/anthropic.claude-code-2.1.291-linux-arm64/…/claude --output-format stream-json --input-format stream-json …`, Agent SDK 0.3.291 |
| Claude Code version | 2.1.291 (extension binary; `/usr/local/bin/claude --version` also reports `2.1.291 (Claude Code)`) |
| Teams flag | `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` (from `~/.claude/settings.json` env) |
| Launch calls (all 5) | `Agent` tool with `name=<role>`, `subagent_type=general-purpose`, background, **no fork**, **no isolation**, no `team_name` |
| Spawn result | "Async agent launched successfully … agentId a…" (no team membership) |
| Completion framing | Harness delivered each result as "[Subagent hand-back] … final report of a subagent this session delegated to"; each produced a task-notification "Agent … finished" |
| Runtime team state for this session (`8509dc6a-…`) | **None.** No `~/.claude/teams/<this session>/config.json`; the five agents are not registered in any team config |
| Contrast | A separate CLI session started 2026-10-06 15:23 in tmux (PID 17830, cwd = this repo, CLI 2.1.291) created `~/.claude/teams/session-e3aa19c3/config.json` with `leadAgentId: team-lead@…`, member `team-lead`, `backendType: "in-process"`. So the **CLI** creates real runtime teams on this machine; this VS Code-extension session did not. |

Conclusion: the five Phase-1 agents ran as **ordinary named subagents**. Their plans are still valid design inputs. Their execution did not satisfy the "real agent team" requirement. All five have stopped (`ListAgents`: none running), so no shutdown is needed and there is no risk of duplicate workers.

## 2. Smallest supported setup change (assessment, not a promise)

**Option A (recommended):** run the lead in the **Claude Code CLI** (`claude`, v2.1.291 at `/usr/local/bin/claude`) from this repo, in VS Code's integrated terminal or tmux, with `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`. Leave `teammateMode` at its default or set it to in-process.
- Supporting evidence: the CLI session already running on this machine created a runtime team config whose lead member has `backendType: "in-process"`. That was observed directly.
- **Not yet verified:** that the CLI session can spawn teammates which register as `members` in that config, or that native shared Task tools appear there. The first bounded check in the new session (§7, step 1) confirms or refutes this before any work is released.

**Option B:** keep the VS Code extension. No evidence suggests it creates runtime teams in this version; not recommended.

No global settings were changed. Nothing was restarted.

⚠️ **Concurrency warning:** a second Claude CLI session (PID 17830, tmux pts/5) is already running in this repo. Use **one** lead session only. Close this VS Code-extension session or leave it idle, so two leads never edit lead-owned files at the same time. That session had modified no project files as of this handoff.

## 3. Artifacts to preserve (do not regenerate)

- `AgentTeamsPrompt.md`: the authoritative spec.
- `src/medquad_qa/contracts/{records,qa,answerability,evaluation,interfaces,__init__}.py`: **contracts v1.1.0**. Verified with `ruff check src tests` (clean), `mypy src/medquad_qa/contracts` (clean) and `pytest tests/contracts` (14 passed).
- `tests/contracts/test_contracts.py`.
- `pyproject.toml` (extras: ml, retrieval, api, ui, tracking, eval, tf, dev; pytest markers gpu, real_model, real_data, docker, slow, integration).
- `constraints/torch-cu130.txt`, `requirements.lock` (193 lines).
- `.venv` with all extras **except tf**. Do NOT reinstall torch; it is `2.14.1+cu130` with CUDA verified.
- `.gitignore`, `.env.example`, `docs/project-status.md` (ledger), `docs/decisions.md` (D-001 to D-018).
- `medquad.csv`: untouched. sha256 `f9fe9e60ea69dbbd5a6987e63d041e54e2048d09e317df830b6ba3387e706d53`.

## 4. Completed checks (do not repeat)

- **Data:** 16,412 rows; columns question/answer/source/focus_area; 5 empty answers; 14 empty focus_area; no URLs or document IDs.
- **Hardware:** GB10 aarch64, sm_121, 121 GiB unified memory. Warm bf16 matmul reaches 85.5 TFLOPS. `nvidia-smi` memory reads N/A.
- **Docker:** Engine 29.6.2, Compose v5.2.0, no sudo needed. GPU access goes through **CDI only**: no nvidia runtime is registered, and the spec is `/var/run/cdi/nvidia.yaml`, generated at runtime. GPU-in-container is **not yet validated**.
- **Installed package versions:** transformers 5.18.0, peft 0.21.2, accelerate 1.15.0, sentence-transformers 6.1.0, qdrant-client 1.19.1, langchain-core 1.6.6, bm25s 0.2.14, PyStemmer 3.1.0, fastapi 0.142.2, uvicorn 0.54.0, streamlit 1.65.0, mlflow 3.16.1, scikit-learn 1.9.1, numpy 2.5.3, pydantic 2.13.5.
- **cuSPARSELt `pip check` line:** closed as benign (D-016). The wheel tag `manylinux2014_sbsa` is unrecognised by pip, but the library is available and a 2:4 sparse matmul ran. Keep it documented.
- **TensorFlow:** 2.21 has an aarch64 cp312 wheel. It is **CPU-only** here (the `[and-cuda]` extra pulls cu12). Not installed yet.

## 5. Approved plans per role (inputs for spawn prompts)

All roles: implement only within owned paths; use synthetic fixtures for tests; record every number from executed commands; report status to the lead, who alone edits the ledger.

### data-steward
- **Owns:** `src/medquad_qa/data/`, `configs/data/`, `scripts/data/`, `tests/data/`, `docs/data/`, `data/`.
- **Modules:** ingest, normalize, ids, question_types, audit, neardup, grouping, splits, leakage, exports, manifests, and a cli (`python -m medquad_qa.data build|audit|split|export|verify`). stdlib csv only.
- **record_id:** `"mq-" + sha256("medquad-rid-v1\x1f"+source+"\x1f"+focus_area_raw+"\x1f"+question_raw+"\x1f"+answer_raw)[:16]`. No row_index in the hash; row_index lives in provenance.
- **Normalization:** NFC, per-line whitespace collapse. Questions also get `" ?"→"?"`. No lowercasing, and no punctuation/number/negation/typo edits. Repeated bullets are flagged, not removed.
- **Exclusions:** 5 empty answers, 48 exact-duplicate copies (mapped to canonical via `Provenance.duplicate_row_indices` and `exclusions.jsonl`), and 21 non-informative answers. About 16,338 records expected; report the actual count.
- **Groups:**
  - `duplicate_group_id`: exact answers, exact pairs and near-dups.
  - `split_group_id`: union-find over the folded topic key across sources, exact normalized questions, non-boilerplate exact answers (boilerplate = shared by ≥3 topics: 21 answers / 533 rows), and near-dups (5-gram masked shingles df≤10, J≥0.8, ≥5 shared rare shingles).
- **Splits:** group-level 80/10/10, stratified by dominant source, seed 20261006. Dry run: 4,688 groups; 13,169 / 1,632 / 1,611 rows.
- **Outputs:**
  - `data/processed/corpus.jsonl`
  - `data/processed/exports/records_{train,val,test}.jsonl`
  - `data/manifests/{source_dataset.json, corpus_manifest.json, split_manifest.jsonl, split_manifest.meta.json, exclusions.jsonl, audit.json, leakage_report.json, exports_manifest.json}`
  - `docs/data/{data_card,audit_report,split_design,reproduction}.md`
- **Versions:** `corpus_version = medquad-1.0.0-<sha12>`; `split_version = split-20261006-<sha12>`.
- **Evaluator pre-freeze items (accepted):**
  - Bracket-stripped topic merges reported as a diagnostic only.
  - Cross-split pairs at J 0.5–0.8 listed by ID.
  - question_type rule version and template table in the meta.
  - Flags `is_boilerplate_answer`, `malformed_question`, `non_informative_answer`.
  - Group counts per split.
- **Acceptance:**
  - 16,412 rows reconcile to corpus plus exclusions.
  - record_ids are unique.
  - Byte-identical rebuild.
  - Zero cross-split overlap on group, topic, question, non-boilerplate answer and J≥0.8.
  - Evaluator sign-off.
  - Synthetic-fixture tests.
- **Disclosures:** the Kaggle licence and slug are unverified, so use locally only and do not redistribute; grouping is weaker than document-level grouping.

### retrieval-engineer
- **Owns:** `src/medquad_qa/retrieval/`, `src/medquad_qa/rag/`, `configs/retrieval/`, `scripts/retrieval/`, `tests/retrieval/`, `tests/rag/`, `docs/retrieval/`, `artifacts/indexes/`. `artifacts/indexes/manifests/` is tracked.
- **BM25:** `bm25s` (k1=1.5, b=0.75) with a deterministic tokenizer. It keeps negations, numbers and aspect words. PyStemmer is optional; 3.1.0 is installed.
- **Dense:** `BAAI/bge-small-en-v1.5` (384-d, with a query prefix), revision pinned at download.
- **Qdrant:** local mode for dev and tests, server mode in Docker (`http://qdrant:6333`).
- **Chunking:** ≤200 words, 40-word overlap, `chunk_id = record_id#c<i>`, char offsets. Chunks aggregate to records by MaxP, with optional collapse of duplicate groups.
- **Index text modes:** `answer` vs `question_answer`, benchmarked separately. Q+A with original questions is reported as **exact-match lookup**.
- **Hybrid:** RRF k=60 over top-50 + top-50. An optional MiniLM cross-encoder is reported as a separate variant.
- **Index version and lifecycle:**
  - `index_version` is content-addressed.
  - Manifest JSON.
  - CLI `build | verify | rebuild | delete-local | list | query | run`.
  - `run` writes per-query ranked hits on frozen queries for the evaluator.
- **Fallback:** `MEDQUAD_RETRIEVER=bm25` is lexical-only (no torch or Qdrant). If Qdrant goes down, hybrid degrades with a `lexical_fallback` warning.
- **RAG chain:** langchain-core only, an explicit LCEL chain: safety rules → retriever → gate → prompt → generator adapter → citation validator.
  - `prompt_version` is `rag-v1+<sha8>`.
  - Evidence goes in `<evidence id="E#">` blocks.
  - Delimiters, chat special tokens, `[E#]` and `mq-` look-alikes are neutralised in evidence and question text.
- **Citations (D-010):** `[E#]` labels map to rewritten `[mq-…]`. Invalid labels are stripped into `invalid_citation_ids`.
- **Abstention order:**
  1. personalized-advice rules (general questions are not refused)
  2. no hits
  3. answerability gate (predictor, else a validation-fitted heuristic)
  4. `INSUFFICIENT_EVIDENCE`
  5. invalid or missing citations
- **Factory:** `medquad_qa.rag.factory.build_pipeline(settings=None, *, observer=None)`, plus a DI constructor that accepts fakes. Readiness keys: corpus, retriever:bm25, retriever:dense, generator:base, generator:finetuned, answerability. Latency keys as in the `QAResponse` docstring.
- **Acceptance:**
  - Protocol conformance; ranks are 1..n; `[]` on no match; the query is treated as data.
  - Chunker invariants; manifest verify detects drift.
  - Unavailable backend → `RetrieverUnavailableError`.
  - Citations ⊆ supplied ⊆ retrieved.
  - Injection fixtures are neutralised.
  - Both personalized-advice directions are tested.
  - A real end-to-end answer after M2.

### model-engineer
- **Owns:** `src/medquad_qa/models/`, `src/medquad_qa/training/`, `configs/models/`, `configs/training/`, `scripts/training/`, `tests/models/`, `tests/training/`, `docs/models/`, `artifacts/models/`.
- **Models (D-011):**
  - `Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554`: Apache-2.0, 8.04 GB, ChatML, non-thinking.
  - Fallback `Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, used only under the switch rule (projected <40% of an epoch within 2.5 h), and the lead is told first.
- **Inference:**
  - `HFGenerator` (bf16 CUDA, SDPA; fp32 CPU smoke path).
  - `max_input_tokens` 3072, raising on overflow.
  - Generator-internal 60 s deadline → `GenerationTimeoutError`.
  - `generate_batch` for evaluation.
  - `load_generator(variant, settings)`: one shared PeftModel, adapter toggled under a lock.
  - `FakeGenerator` for tests.
- **Matched generation (D-012):** greedy, max_new_tokens 256, repetition_penalty 1.05.
- **SFT (D-013):** closed-book, train split only. Loss is masked to assistant tokens. Max sequence length 1024; truncation stops at a sentence boundary with **no EOS**. Truncation counts are reported per source.
- **LoRA:** r16/α32/dropout 0.05 on all linear layers; lr 2e-4 cosine, 3% warmup; batch 4×4; bf16 with gradient checkpointing; 1 epoch; seed 42; hard cap 2.5 h.
- **Run order and logging:**
  - A 20-step smoke run first, which sets max_steps.
  - MLflow logging, with fallback `file:artifacts/models/mlruns-local`.
  - The adapter is reloaded in a fresh process and generates real responses.
  - A RAG-formatted SFT fraction is an optional separate run only.
- **Answerability:** pairs built within each split (agreed with the evaluator).
  - Positive: same topic and question type, excluding boilerplate.
  - Negatives, about 1:1: easy, same-topic-different-type, and BM25-hard drawn from train only.
  - Uses the `AnswerabilityPair` contract.
- **Classifiers:**
  - Baseline: lexical features with sklearn LogisticRegression.
  - Keras 3 (TF CPU, `[tf]` extra or `.venv-tf` per D-014): a BiGRU siamese model plus lexical features.
  - Threshold = max-F1 on validation (D-018).
  - Metrics: ROC-AUC, PR-AUC, Brier, ECE (10 equal-width bins), reliability plot.
  - No TF in the API image.
- **Acceptance:**
  - Offline tests on a tiny random-init Qwen config.
  - A GPU real-generation test.
  - Bench JSON with load time, tok/s and peak memory.
  - A run manifest (seeds, hyperparameters, data sha256s, revision, env, runtime, memory).
  - Adapter reload evidence.
  - Classifier metrics.
  - Model cards.

### service-platform-engineer
- **Owns:** `src/medquad_qa/api/`, `src/medquad_qa/ui/`, `src/medquad_qa/observability/`, `configs/service/`, `configs/monitoring/`, `scripts/service/`, `scripts/ops/`, `tests/api/`, `tests/observability/`, `deploy/`, `.github/workflows/`, `docs/service/`, `docs/operations/`, `mlruns/`, `artifacts/service/`.
- **API:** `create_app(settings=None, pipeline_provider=None)`.
  - The lifespan builds the pipeline in the background; if that fails, live stays 200 and ready returns 503.
  - Middleware: X-Request-ID sanitised to `REQUEST_ID_PATTERN`, and a 16 KB body limit returning 413.
  - Error mapping: 503 (artifact or mode unavailable, busy) with Retry-After; 504 on timeout; 502 on GenerationError; 500 otherwise. The 422 body never echoes input.
  - One global generation semaphore and a 120 s asyncio backstop.
- **Endpoints:** `POST /v1/qa`, `GET /health/live`, `GET /health/ready`, `GET /metrics`, and an optional `GET /v1/info`.
- **Logging:** JSON logs with **no question or answer text by default**. `MEDQUAD_LOG_CONTENT=true` logs a hash plus 200 chars, with a startup warning.
- **Metrics:** prefix `medquad_`, per the plan's catalogue:
  - http requests, durations and inflight
  - qa requests, errors and latency, plus component latency
  - abstentions and citation failures
  - retrieval score, input length
  - artifact, build and component_ready info
  - offline_eval gauges
- **UI:** Streamlit over HTTP only. It shows a disclaimer, explicit mode labels, citations, an invalid-citation warning and versions. Usage stats are off.
- **Compose:** `deploy/compose.yaml` with qdrant, mlflow (sqlite, 127.0.0.1:5000), api-gpu (profile gpu), api-cpu (profile cpu), ui, and prometheus (profile monitoring).
  - Everything binds to 127.0.0.1 and runs as non-root uid 10001 with a read-only rootfs, `cap_drop ALL` and `no-new-privileges`.
  - Read-only mounts for artifacts, data and the HF cache, with `HF_HUB_OFFLINE=1`.
  - GPU via CDI `devices: [nvidia.com/gpu=all]`; Compose syntax to be verified.
  - Dockerfiles in `deploy/docker/` with per-Dockerfile ignore files.
- **GPU decision (D-017):** build the GPU image only if `scripts/ops/validate_gpu_container.sh` passes; otherwise the API runs on the host (option B).
- **CI:** `.github/workflows/ci.yml` runs lint, mypy, offline pytest (x86 + arm), compose config check, CPU image build (no push) and pip-audit (non-blocking). There is no remote, so a local `make ci` is the evidence.
- **Make targets** (lead's Makefile wraps the service scripts): up, down, gpu-check, stack-smoke, ci, api-host.
- **Acceptance:** the user's six-step Docker sequence (compose config → build+up → live/ready → real `POST /v1/qa` → `/metrics` + `docker compose logs` → commands and results to the lead) plus a rollback drill.

### evaluation-safety-engineer
- **Owns:** `src/medquad_qa/evaluation/`, `configs/evaluation/`, `scripts/evaluation/`, `tests/evaluation/`, `tests/security/`, `tests/integration/`, `docs/evaluation/`, `docs/security/`, `artifacts/evaluation/`.
- **Track A:** 300 TEST queries, at most 1 per test group.
  - 200 template paraphrases (`synthetic_rule`) and 100 agent-written (`llm_generated_unreviewed`).
  - Gold = all records with the same topic and question type.
  - No exact or normalized match to any indexed question.
- **DEV:** 60 queries from val groups. This is the only set allowed for tuning.
- **Track B:** the same 300 queries in all 4 modes, plus a 100-query train-group memorization probe. RAG on held-out groups is reported as retrieval-assisted answering.
- **Track C:** about 240 items: answerable, unanswerable, hard-negative, personalized advice with general controls, ambiguous, and conflicting (via a FixtureRetriever).
- **Leakage:** `check_leakage` fails the build if eval overlaps SFT, classifier or threshold data. Thresholds and prompts are frozen before any TEST run.
- **Metrics:**
  - Recall@k, MRR (nDCG only when graded labels exist)
  - citation validity, coverage and support (NLI ≤0.7 GB)
  - unsupported-claim rate
  - correctness and completeness rubric (0–2 each)
  - abstention confusion matrix and over-refusal
  - classifier AUROC, AUPRC, Brier and ECE
  - latency median and p95 (200 requests per mode, plus 100 through the Docker API)
  - ROUGE-L and token-F1 as supplementary only
- **Judge:** no 7B judge (D-011). An optional LLM judge is provisional, and 50 judged items get an AI-agent check that is **not** labelled human.
- **Statistics:** paired cluster bootstrap (10k resamples) and exact McNemar.
- **Spot-check:** the user answered "maybe later". Export the 50-item sheet; only items the user actually fills in become `human_reviewed`.
- **Security tests:** injection in evidence, citation integrity, personalized advice, input bounds, missing artifacts (503), timeout (504), secret scan, logging privacy. Findings go to `docs/security/findings.md` with severity, reproduction, files, evidence, fix and retest.

## 6. Task dependencies (unchanged from the ledger)

- G1 (released only once a real team is confirmed) unlocks D2, R2, M2, S2, E3.
- D2 → D3 → D4 (with evaluator sign-off) → D5.
- D5 → M3 → M4, and D5 → M5.
- D4 → E2.
- R2 → R3 → R4. R2 + M2 → R5 → R6.
- S2 → S3/S5. S2 + Docker → S4. S4 + R5 + M2 → S6.
- E2 + E3 + R4 + R5 + M4 + M5 → E4. R5 + S2 → E5. E4 + E5 → E6.
- Lead: L2 is still to finish (Makefile, CLAUDE.md, architecture overview), then L4 and L5.

## 7. Exact instructions for the new lead session

1. **Bounded team check** (no implementation):
   - Start `claude` (CLI 2.1.291) in this repo, in the VS Code integrated terminal or tmux, with only one lead session active.
   - Confirm `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`.
   - Spawn **one** teammate, data-steward, with the role prompt from §5 and a "read this handoff; do not edit yet" instruction.
   - Verify that `~/.claude/teams/<session>/config.json` lists it under `members`, and check whether native Task tools exist.
   - If it does not register as a member, stop and report to the user.
2. If it is confirmed, spawn the remaining four with prompts built from `AgentTeamsPrompt.md` §7–11 plus §5 above. **Do not re-run discovery.** Plans are approved as written, subject to D-009 to D-018.
3. Send **START IMPLEMENTATION** to each, including:
   - the approved plan
   - contracts v1.1.0
   - ownership
   - dependencies (already installed; tf pending)
   - gate rulings
   - acceptance criteria
4. Use native shared Task tools if they are present. Otherwise keep `docs/project-status.md` lead-owned, with status changes reported by message.
5. Keep the original constraints throughout:
   - Moderate disk budget.
   - No push, publishing, paid compute or public deployment.
   - No Docker socket changes and no sudo.
   - No blind GPU package reinstall.
