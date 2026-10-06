# MedQuAD Evidence-Grounded Medical QA

A non-clinical research prototype for medical information. The full spec is in `AgentTeamsPrompt.md`. Live status is in `docs/project-status.md`, and decisions are in `docs/decisions.md`.

## Hard rules
- Never imply clinical validation, patient-outcome prediction, publication, or employment history. Answerability scores and retrieval scores are not medical confidence.
- Never invent metrics, labels, training runs, reviews, or passed tests. Every number must trace back to an executed command and a saved artifact.
- Do not modify `medquad.csv`. Never commit raw data, credentials, vector indexes, model weights, or query logs.
- Do not push, publish, deploy publicly, or use paid compute. Do not use sudo or change the Docker socket. Bind services to 127.0.0.1.
- Do not reinstall torch. It is `2.14.1+cu130`, pinned by `constraints/torch-cu130.txt`. Install with `-c constraints/torch-cu130.txt`.
- Only label an item `human_reviewed` after the user has actually reviewed it.

## Ownership
Every path has exactly one owner (`AgentTeamsPrompt.md` §5; additions in D-001 and D-015). The lead owns the contracts (`src/medquad_qa/contracts/`), the root config, `docs/architecture/`, `docs/project-status.md`, `docs/decisions.md`, and `docs/portfolio/`. Only the lead runs `git commit`. To change a file you don't own, message its owner.

## Commands
- `make test`: offline tests (excludes the gpu, real_model, docker, and slow markers).
- `make lint` / `make typecheck` / `make ci`
- See `Makefile` for the data, index, training, eval, and stack targets.

## Environment
- Hardware: GB10, aarch64, CUDA 13, 121 GiB of unified memory.
- Python: `.venv` running Python 3.12.3.
- Docker: Engine 29.6.2 with Compose v5.2.0. In-container GPU access works only through CDI, and only if `make gpu-check` passes.
