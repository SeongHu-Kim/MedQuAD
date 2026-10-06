# Project Status — MedQuAD Evidence-Grounded Medical QA

Owner: lead. Status categories: **COMPLETE** (implemented, executed, validated) ·
**PARTIAL** (implemented, missing execution/validation) · **BLOCKED** · **PENDING** (not started).

Last updated: 2026-10-06 (session 2, lead)

## Session log

### Session 1 — 2026-10-06 — Phase 0 discovery (lead only; no teammates spawned)

Discovery results (executed commands, outputs summarized):

| Check | Command | Result |
|---|---|---|
| Repo contents | `ls -la` | Only `medquad.csv`, `AgentTeamsPrompt.md`. No CLAUDE.md, no `agent-teams-guide.md`, no job posting, no tests, not a git repo. |
| Dataset checksum | `sha256sum medquad.csv` | `f9fe9e60ea69dbbd5a6987e63d041e54e2048d09e317df830b6ba3387e706d53` (22,835,609 bytes, mtime 2024-09-07) |
| Dataset schema | csv profile (python stdlib) | 16,412 rows; columns `question, answer, source, focus_area`; all rows 4 fields |
| Missingness | same | `answer` empty: 5 · `focus_area` empty: 14 · `question`/`source` empty: 0 |
| Uniqueness | same | unique questions 14,984 · unique answers 15,818 · sources 9 · focus areas 5,127 |
| Source counts | same | GHR 5430, GARD 5394, NIDDK 1192, NINDS 1088, MPlusHealthTopics 981, NIHSeniorHealth 769, CancerGov 729, NHLBI 559, CDC 270 |
| Metadata gaps | — | **No source URLs, document IDs, or question-type column.** Kaggle slug/version unverified. Grouping must use focus_area/source/content hashes (weaker leakage control — to be documented). |
| OS / arch | `uname -a` | Ubuntu 24.04.5, Linux 7.0.0-1019-nvidia, **aarch64** |
| CPU / RAM / disk | `nproc`, `free -h`, `df -h` | 20 CPUs · 121 GiB unified memory · 3.4 TB free |
| GPU | `nvidia-smi` | NVIDIA GB10, driver 580.178.04, CUDA 13.0 |
| Python | `python3 --version` | 3.12.3; project venv `.venv/` created |
| ML packages | `pip list` | None preinstalled. PyTorch `2.14.1+cu130` installed in `.venv`; CUDA available on GB10 (see Background PyTorch install) |
| Agent teams | env | `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`; named teammates + SendMessage available; **no shared Task* tools** → this file is the dependency ledger |
| Docker | `docker info` | **BLOCKED in this session**: `permission denied ... /var/run/docker.sock`. User added docker group and verified in their own terminal (Engine 29.6.2, Compose v5.2.0), but this Claude process (`id`) lacks the `docker` group because it started before the change. `docker compose version` → v5.2.0. **Requires Claude Code session restart.** No socket permission changes or sudo used. |
| Kaggle credentials | `ls ~/.kaggle` | None (not needed: CSV is local) |

User decisions (session 1):
- Resource budget: **Moderate** — ~1–4B instruct model + small embedding model, ~30–40 GB disk, LoRA run ≤ ~2–3 h on GB10. Local only, no paid compute.
- Docker: user fixed group access; validate stack after session restart. Use ARM64 images; validate GPU-in-container separately before relying on it.
- Git: `git init` done, local commits allowed (lead coordinates), **no remote push**.
- Venv + PyPI/HF Hub installs allowed within budget.
- Job posting unavailable → skill matrix uses the skill list from `AgentTeamsPrompt.md` §1, with that limitation stated.

Lead artifacts written in session 1:
- `src/medquad_qa/contracts/{records,qa,evaluation,interfaces,__init__}.py` (contracts v1.0.0, **draft — not yet reviewed with teammates, not yet import-tested**)
- `src/medquad_qa/__init__.py`, package/ownership directory skeleton
- `.gitignore`, `.env.example`, `docs/project-status.md`, `docs/decisions.md`

### Session 2 — 2026-10-06 — resume, team spawn for Phase 1 planning

| Check | Command | Result |
|---|---|---|
| Docker group | `id -nG` | includes `docker` |
| Docker daemon | `docker info --format ...` | Engine 29.6.2, arch aarch64, CDI dirs `/etc/cdi`, `/var/run/cdi` |
| Compose | `docker compose version` | v5.2.0 |
| PyTorch | `.venv/bin/python -c "import torch; ..."` | `2.14.1+cu130 True` (no reinstall) |
| Agent-teams flag | `printf ... $CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS` | `1` (from `~/.claude/settings.json`; no conflicting project/managed settings) |
| Core deps | `.venv/bin/pip install -e ".[dev]"` | pydantic, numpy, pandas, pyarrow, pyyaml, pytest, ruff, mypy installed |
| Contracts | import + validation smoke, `ruff check src`, `mypy src/medquad_qa/contracts` | import OK (33 exports, v1.0.0); bound check raises ValidationError; ruff clean; mypy clean |

Team spawn (Agent tool, named, `general-purpose`, background, read-only planning phase):
data-steward, retrieval-engineer, model-engineer, service-platform-engineer, evaluation-safety-engineer — all five launched and listed as **running** by `ListAgents`.
**Caveat:** `ListAgents` classifies all five as **"Subagents"**, not as agent-team teammates. This session exposes no TeamCreate or shared Task* tools, so formal agent-team mode is **not verified**. They are named, individually addressable via SendMessage, and receive the full role prompts. User informed; awaiting decision on whether this substitute is acceptable.

Open issues:
- cuSPARSELt: `pip check` → `nvidia-cusparselt-cu13 0.8.1 is not supported on this platform`. **OPEN**; model-engineer asked to assess in plan.
- GPU-in-container: not yet validated.

### Session 2 (cont.) — Architecture gate G1 review

- All five Phase-1 plans received (data-steward, retrieval-engineer, model-engineer, service-platform-engineer, evaluation-safety-engineer). All were read-only; none changed files.
- Contracts **v1.1.0** written (decision D-009). Checks run: `ruff check src tests`, `mypy src/medquad_qa/contracts`, `pytest tests/contracts`. Ruff and mypy are clean, and pytest shows 14 passed.
- Dependencies installed: `pip install -c constraints/torch-cu130.txt -e ".[ml,retrieval,api,ui,tracking,eval,dev]"` exited 0.
  - torch stayed `2.14.1+cu130`, CUDA is still available, and the main imports work.
  - Versions: transformers 5.18.0, peft 0.21.2, sentence-transformers 6.1.0, qdrant-client 1.19.1, langchain-core 1.6.6, bm25s 0.2.14, PyStemmer 3.1.0, fastapi 0.142.2, streamlit 1.65.0, mlflow 3.16.1, scikit-learn 1.9.1.
  - `pip check` only flags the cuSPARSELt platform tag, which is harmless (D-016).
  - Log: `artifacts/logs/deps_install.log`. Lockfile: `requirements.lock` (193 lines). The TensorFlow extra is not installed yet; model-engineer will handle it at M5.
- Gate rulings: D-009 to D-018. **START IMPLEMENTATION has not been sent yet.** It waits on the user decision about named subagents vs. agent-team teammates (D-007).

### Session 2 (cont.) — Execution-mode verification → implementation PAUSED

The user required a real agent team; START IMPLEMENTATION was withheld. A bounded verification found the following:
- **Interface:** VS Code extension (`CLAUDE_CODE_ENTRYPOINT=claude-vscode`, binary in `anthropic.claude-code-2.1.291-linux-arm64`, Agent SDK 0.3.291). Claude Code is 2.1.291, and the teams flag is 1.
- **The five agents were spawned as ordinary subagents.** They were launched via `Agent(name=…, subagent_type=general-purpose)` with no fork and no isolation. The spawn result read "Async agent launched successfully". Their results came back as "[Subagent hand-back]". **This session has no runtime team config**, and the five agents are not team members.
- **For comparison,** another CLI session (PID 17830, tmux, started 15:23, same repo) created `~/.claude/teams/session-e3aa19c3/config.json` with lead `backendType: in-process`.
- **Result: CONFIRMED ordinary subagents.** All five have stopped (`ListAgents`: none running), and no implementation was released.
- **Migration handoff:** `docs/architecture/team-migration-handoff.md`. It contains the approved plans, contracts, checks, dependencies and continuation steps.

### Session 3 — 2026-10-06 — CLI lead, genuine agent team confirmed

Current verified status. The "BLOCKED" Docker entries in the session-1 log above are historical.
- Interface: `CLAUDE_CODE_ENTRYPOINT=cli`, Claude Code 2.1.291, run in tmux as `claude --resume 8509dc6a-… --teammate-mode in-process` (PID 20823). Teams flag is 1.
- Docker: `id -nG` includes docker; `docker info` reports Engine 29.6.2 on aarch64; Compose is v5.2.0.
- Team: this session owns `~/.claude/teams/session-412b5373/config.json`, which lists `team-lead@session-412b5373` as lead. Its `createdAt` matches the process start time, 15:48:51.
- Registration test: after `Agent(name="data-steward")`, the spawn result said "will receive instructions via mailbox". The config then listed `data-steward@session-412b5373` with backendType `in-process`. The teammate sent a handoff acknowledgment with "no material discrepancies" and then an `idle_notification`. **Result: genuine teammate confirmed.**
- Native Task tools: none in this session either (ToolSearch for TaskCreate, TaskList and similar found no matches). `docs/project-status.md` stays the lead-owned ledger.
- Note: an older CLI session (PID 17830, pts/5, started 15:23) was still running. Its team `session-e3aa19c3` has no teammates. The user was asked to exit it.
- data-steward alignment notes accepted by the lead:
  - quality flags use bare names such as `boilerplate_answer`;
  - `non_informative_answer` appears only as an exclusion reason;
  - split and corpus_version live in the manifests;
  - the build asserts that duplicate collapse keeps IDs unique and reports the actual count;
  - PyYAML is used for the config files.

### Session 3 (cont.) — Gate G1 released

- All five genuine teammates acknowledged the handoff.
  - data-steward, model-engineer, service-platform-engineer: no material discrepancies.
  - evaluation-safety-engineer: 1 material gap, fixed by contracts v1.1.1 (D-021).
  - retrieval-engineer: 2 material points, ruled on as D-022 (evidence budget) and D-023 (safety rules in all modes).
- Contracts v1.1.1: `ruff`, `mypy` and `pytest tests/contracts` all pass (14 passed).
- Baseline commit `b24cf70` (local only; no dataset, venv, or .env).
- **START IMPLEMENTATION sent to all five at 2026-10-06 ~16:00.**
- Teammates are not committing. The lead commits after reviewing evidence.

- **S2 verified by the lead:**
  - `pytest tests/api tests/observability`: 62 passed.
  - `ruff check` and `ruff format --check` on S2 paths: clean.
  - `mypy src/medquad_qa/api src/medquad_qa/observability`: clean (12 files).
  - Teammate's host smoke run: live returned 200 and ready returned 503. The 503 is expected, because `rag.factory` is not built yet, and the app reports it as a readiness failure rather than crashing.
  - Known: running `mypy` over the tests directories in a single call reports a duplicate `conftest` module. Running them separately passes.

- **D2 and D3 verified by the lead.**
  - `python -m medquad_qa.data verify` reports "12 files byte-identical; leakage checks passed".
  - `pytest tests/data` shows 32 passed. Ruff and mypy are clean.
  - The corpus has 16,336 unique record_ids, and all of them validate against `MedicalRecord`.
  - An answer-text probe found 0 of 278 sampled snippets in the tracked `data/manifests`.
  - Versions: corpus_version `medquad-1.0.0-34d97c16127f`, split_version `split-20261006-c759a1668f89`.
  - The CSV's 16,412 rows reconcile to 16,336 records plus 76 exclusions (5 empty, 48 duplicate copies, 23 non-informative).
  - Splits in records: 13,021 train, 1,648 validation, 1,667 test. Split groups: 3,464 / 471 / 390.
  - The teammate also ran an independent second build, which produced identical sha256 for all files.

- **E3 verified by the lead:** `pytest tests/evaluation -m "not real_model"` shows 38 passed, and ruff and mypy are clean.
  - NLI model: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli@6f5cf0a2…` (MIT, 369 MB, sha256 matches the HF LFS oid). The real_model NLI smoke test passed.
  - The model-engineer real-generator tests are marked `real_model`/`gpu` and are excluded from `make test`.
- **Open:** two cross-split topics differ only by a bracketed name ("bile duct cancer (cholangiocarcinoma)", "chronic fatigue syndrome (cfs)"). This is pending the evaluator's ruling during E1 sign-off.

- **R2/R3/R5 verified by the lead:**
  - `pytest tests/retrieval tests/rag -m "not real_model"`: 99 passed.
  - ruff and mypy are clean on 26 files.
  - The embedder is `BAAI/bge-small-en-v1.5@5c38ec7c…` (MIT).
- **Incident:** a bulk `ruff format` run by model-engineer outside its own paths turned `\u` escapes in `rag/sanitize.py` into literal bidi control characters.
  - retrieval-engineer restored the escapes.
  - The lead added `tests/test_repo_hygiene.py`, which fails if invisible or control characters appear in `src/`.
  - model-engineer was reminded of the ownership rule.

## Resume checklist (session 2)

1. `id | grep docker && docker info && docker compose version` — must succeed; else stop and report.
2. Verify torch (see "Background PyTorch install" below): `cat artifacts/logs/torch_install.log`; `.venv/bin/pip check`; `.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`. If the import fails or the package is partial (install interrupted by exit), reinstall with the same command; only then proceed.
3. Lead: write `pyproject.toml`, `Makefile`, `CLAUDE.md`, `docs/architecture/{overview,contracts}.md`; install base deps; import-test contracts; first commit.
4. Spawn the five teammates (prompts per §Task ledger and `AgentTeamsPrompt.md` §7–11); collect plans; send **START IMPLEMENTATION** after contract review.
5. Docker validation sequence for service-platform-engineer (user request, 2026-10-06): compose config validate → build + up → `/health/live`, `/health/ready` → real `POST /v1/qa` with real artifacts → `/metrics` + `docker compose logs` → record commands/results here. Validate GPU-in-container (`docker run --rm --device nvidia.com/gpu=all <arm64 cuda image> nvidia-smi`) before any GPU container workload.


### Background PyTorch install (state at end of session 1)

- Status: **COMPLETE (exit code 0)**. Installed `torch 2.14.1+cu130` (`triton 3.8.0`, CUDA 13 wheels). Verification printed `2.14.1+cu130 True NVIDIA GB10`, and a 512×512 CUDA matmul ran (sum -8357.09). Warnings: NumPy not installed yet (it comes with the base deps); `pip check` reports `nvidia-cusparselt-cu13 0.8.1 is not supported on this platform`, which did not affect the matmul check. Recheck it when the LLM workloads run.
- Process: pip PID `23621` (finished). Command ran as a child of Claude Code's shell.
- Full command: `.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu130 torch 2>&1 | tail -3`, then a CUDA matmul check (`torch.__version__`, `cuda.is_available()`, device name, 512×512 matmul).
- Log: `artifacts/logs/torch_install.log`. It is **empty until the command finishes**, because output is piped through `tail -3`. Claude task output copy: `/tmp/claude-1000/-home-hangds-Desktop-SeongHuKim-MedQuAD/8509dc6a-7edf-42b5-bc59-cbc01100fbfa/tasks/bcnhott7m.output`.
- Check from any terminal: `ps -p 23621 -o pid,etime,cmd` (no row means it has finished or was killed).

## Task ledger

Gate: G1 = architecture gate (all plans reviewed + lead sends START IMPLEMENTATION).
Status of every task below: **PENDING** (team not yet spawned).

### data-steward (owns src/medquad_qa/data/, configs/data/, scripts/data/, tests/data/, docs/data/, data/)
| ID | Task | Depends | Acceptance |
|---|---|---|---|
| D1 (COMPLETE: plan received) | Discovery: schema, provenance, license/usage restrictions, plan to lead | — | plan message received |
| D2 (COMPLETE) | Ingestion + schema mapping → MedicalRecord (raw+normalized) | G1 | real CSV → corpus.jsonl, row count reconciles to 16,412 minus documented exclusions |
| D3 (COMPLETE) | Quality audit: missing, malformed, exact/near dups, lengths | D2 | `docs/data/audit_report.md` with actual counts from saved JSON |
| D4 (COMPLETE: frozen split-20261006-2f0fb25ee6d8, evaluator signed off (D-034)) | Grouping (duplicate + split groups) + deterministic splits + leakage checks | D3, E1 review | split manifest + leakage check script passes; evaluator sign-off |
| D5 (COMPLETE: exports on the frozen split, sha256 in exports_manifest) | Training/eval exports for model-engineer & evaluator | D4 | export files + checksums in manifest |
| D6 (COMPLETE: data card with disclosures, split design, reproduction; 44 tests) | Tests, data card, reproduction commands | D2–D5 | pytest tests/data green; data card cites artifacts |

### retrieval-engineer (owns retrieval/, rag/, configs/retrieval/, scripts/retrieval/, tests/retrieval/, tests/rag/, docs/retrieval/, artifacts/indexes/)
| ID | Task | Depends | Acceptance |
|---|---|---|---|
| R1 (COMPLETE: plan received) | Discovery + retriever/RAG design plan | — | plan received |
| R2 (COMPLETE: real BM25 indexes bm25-answer-3c6bf6f998cf, bm25-qa-d5f777b26f79) | BM25 retriever (common interface), lexical fallback | G1, fixtures | unit tests on fixtures; runs on real corpus after D2 |
| R3 (COMPLETE on local mode: 4 real indexes on corpus 86e384302357, 16,336 records / 26,596 chunks, verify ok; server-mode Qdrant build pending S4) | Dense retriever + Qdrant index lifecycle + index manifest | R2, D2 | build/rebuild/verify commands; manifest w/ model rev + corpus_version |
| R4 | Hybrid (RRF) + optional reranker; answer-only vs Q+A indexing variants | R3 | all variants run on frozen eval queries |
| R5 (PARTIAL: LCEL pipeline, safety, sanitiser, citations, gate, budget implemented offline; real generator pending M2) | LangChain RAG pipeline, prompt templates, citation validation, abstention | R2, M1 | invalid IDs stripped+flagged; injection-in-evidence tests |
| R6 | Tests, failure cases, docs; respond to E5 review | R2–R5 | tests green; findings resolved/documented |

### model-engineer (owns models/, training/, configs/models/, configs/training/, scripts/training/, tests/models/, tests/training/, docs/models/, artifacts/models/)
| ID | Task | Depends | Acceptance |
|---|---|---|---|
| M1 (COMPLETE: plan received) | Discovery: model selection within Moderate budget (ARM64/CUDA13 compat), plan | — | plan w/ model id+revision+license, disk/runtime estimate |
| M2 (COMPLETE: real CUDA bf16 and CPU generation tests pass; bench: load 5.9 s, ~19.9 tok/s single, ~150 tok/s at batch 8, peak 8.2 GiB CUDA) | Base inference adapter (Generator) | G1 | real generation on GPU + CPU-path smoke |
| M3 (PARTIAL: pipeline + CPU tiny-model smoke done; waiting for D5 exports) | SFT export from approved train split; prompt-masked LoRA pipeline | D5 | smoke run (few steps) executes |
| M4 | Full LoRA run + reload + generation test; MLflow logging | M3 | run manifest, loss curves, adapter reload test |
| M5 | Answerability: statistical baseline + TF/Keras classifier, hard negatives, val-chosen threshold, calibration | D5, E1 label design | metrics JSON on val/test; calibration plot |
| M6 | Model cards, run manifests, tests | M2–M5 | pytest green; cards cite artifacts |

### service-platform-engineer (owns api/, ui/, observability/, configs/service/, configs/monitoring/, scripts/service/, scripts/ops/, tests/api/, tests/observability/, deploy/, .github/workflows/, docs/service/, docs/operations/)
| ID | Task | Depends | Acceptance |
|---|---|---|---|
| S1 (COMPLETE: plan received) | Discovery + service/deploy plan (ARM64 images) | — | plan received |
| S2 (COMPLETE: offline with FakePipeline) | FastAPI: /v1/qa, /health/live, /health/ready, /metrics; bounds, timeouts, request IDs | G1 | API tests with fake pipeline green |
| S3 (PARTIAL: Streamlit UI and AppTest tests done; manual smoke against a real answer pending GPU) | Streamlit demo (answer, citations, versions, disclaimer) | S2 | manual smoke against running API |
| S4 (PARTIAL: hardened arm64 compose, CPU stack up and healthy on 127.0.0.1; GPU image approved, build pending) | Dockerfiles + Compose (api, ui, qdrant, mlflow; localhost binds, non-root) | S2, **Docker access** | compose config validates; stack up; health OK |
| S5 (COMPLETE: metrics, logs, MLflow server smoke, Prometheus scrape up=1, CI workflow (never run on GitHub); local make ci exit 0) | Metrics, structured logs (no raw content by default), MLflow server, CI workflow | S2 | metrics test; CI lint+offline tests locally |
| S6 | Real-artifact integration in stack + runbook (deploy, rollback, troubleshooting) | S4, R5, M2 | real QA request recorded here |

### evaluation-safety-engineer (owns evaluation/, configs/evaluation/, scripts/evaluation/, tests/evaluation/, tests/security/, tests/integration/, docs/evaluation/, docs/security/, artifacts/evaluation/)
| ID | Task | Depends | Acceptance |
|---|---|---|---|
| E1 (COMPLETE: F-001 retest PASS; split sign-off granted) | Eval tracks, leakage controls, answerability label design; review D4 split design | — | written approval/requests to data-steward & model-engineer |
| E2 (COMPLETE: 700 frozen items, A 300 / C 240 / probe 100 / dev 60, sha256-pinned, leakage 0 errors; spot-check sheet exported, non-blocking) | Frozen eval set (paraphrases, hard negatives, case types) with label provenance | D4 | manifest w/ checksum; no test-set tuning |
| E3 (COMPLETE: harness + fixtures + threat model, 38 offline tests) | Metric harness: retrieval, citation validity/support, rubric, abstention, classifier, latency | G1 | unit tests on fixtures |
| E4 | Run 4-way comparison (base / rag / finetuned / finetuned_rag) + retriever comparison | E2, E3, R5, M4 | saved metrics + failure examples |
| E5 (PARTIAL: first pass done, 72 passed, open findings marked strict-xfail; F-002 high open) | Security/safety tests: injection, citations, personalized advice, malformed input, missing artifacts, secrets, logging | R5, S2 | severity-ranked findings + retests |
| E6 | Independent evaluation report | E4, E5 | report citing artifacts |

### lead
| ID | Task | Depends | Acceptance |
|---|---|---|---|
| L1 | Phase 0 discovery | — | **COMPLETE** (this file) |
| L2 | Contracts, pyproject/lock, Makefile, CLAUDE.md, architecture docs | L1 | **COMPLETE**: contracts v1.1.1 (14 tests pass), requirements.lock, Makefile (`make test` runs), CLAUDE.md, docs/architecture/{overview,contracts}.md. Stack targets wait on service-platform-engineer scripts |
| L3 | Spawn team, review plans, START IMPLEMENTATION | L2 | **COMPLETE**: genuine team `session-412b5373` with 5 in-process members; acknowledgments received; G1 released (D-020 to D-024) |
| L4 | Integration arbitration, commits, decision log | L3 | — |
| L5 | Portfolio: README, diagram, skill matrix, tech report, demo script, talking points | E6 | — |
