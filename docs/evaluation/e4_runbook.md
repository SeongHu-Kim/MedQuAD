# E4 runbook: frozen four-mode evaluation run

Owner: evaluation-safety-engineer. Decisions: D-035 (held-out), D-037/D-042/D-043 (frozen config), D-045 (run alone), D-048 (plan).

## What runs
`scripts/evaluation/e4_full.sh` runs one item at a time (batch-1) in this priority order. Output goes to `artifacts/evaluation/e4/runs/`.

| Tier | Set | Modes | Items |
|---|---|---|---|
| 1 | `test_track_a` | base, rag, finetuned, finetuned_rag | 300 × 4 |
| 2 | `train_probe` | base, finetuned | 100 × 2 |
| 3 | `test_track_c` | rag, finetuned_rag | 240 × 2 |
| 4 | `test_track_c` personal-advice + general controls | base, finetuned | 80 × 2 |

- `scripts/evaluation/e4_env.sh` pins the environment and the expected runtime versions:
  - safety rules, RAG prompt, closed-book prompt;
  - retriever and index, answerability gate;
  - base and fine-tuned `model_version`.
- `run_e4_pipeline.py` refuses to start (exit 2) on any version mismatch. It also refuses (exit 3) if any retriever component is degraded, so no silent BM25 fallback can occur.
- Step 0 (`scripts/evaluation/e4_step0.sh`) is the 15-item DEV check run before the TEST run.

## Preconditions (check before launch)
- Exactly one evaluation process is running. No other medquad pipeline, API or GPU job, and no Docker stack (D-045):
  ```
  pgrep -af "run_e4_pipeline|medquad_qa" ; nvidia-smi ; docker ps
  ```
- There is free disk space: `df -h /`.
- The frozen inputs are unchanged. `load_eval_set` checks every eval set's sha256 against its manifest, and each run's `*.meta.json` records the sha256s.

## Launch (detached; independent of any chat or agent session)
```
tmux new-session -d -s medquad-e4 'cd /home/hangds/Desktop/SeongHuKim/MedQuAD && TIERS="1 2 3 4" bash scripts/evaluation/e4_full.sh 2>&1 | tee -a artifacts/logs/e4_full_$(date -u +%Y%m%dT%H%M%SZ).log'
```
If you want to stop after tier 3 (D-048), use `TIERS="1 2 3"`.

**Actual launch (2026-10-06T12:52:28Z, D-051).** The lead ran it in tmux session `medquad-e4` from the repo root. It used a no-`tee` form so that the script's real exit code is recorded; with `| tee`, the pipeline's status would be tee's:
```
LOG=artifacts/logs/e4_full_20261006T125228Z.log
TIERS="1 2 3 4" bash scripts/evaluation/e4_full.sh >"$LOG" 2>&1; echo "E4_FULL_EXIT=$? <utc>" >>"$LOG"
```
The run is finished when `E4_FULL_EXIT=0` appears in the log. A non-zero value means the run stopped early. Report the value first, then resume with the same command.

## Check progress
- To attach, run `tmux attach -t medquad-e4`. To detach without stopping the run, press `Ctrl-b d`.
- To follow the log, run `tail -f artifacts/logs/e4_full_*.log`. It prints one `<mode> done …` line per set×mode and one `TIER_<n>_DONE <utc>` line per tier.
- To count items done, run `wc -l artifacts/evaluation/e4/runs/per_item/*.jsonl`. There is one line per answered item, flushed after every item.
- Each finished set×mode writes `artifacts/evaluation/e4/runs/<set>__<mode>.meta.json`. It records:
  - git sha, versions, readiness, environment and platform;
  - eval-set and manifest sha256s;
  - resumed, new and total row counts, and errors;
  - wall time.

## Resume after an interruption
Rerun the same launch command. It is safe to repeat.
- Resume works by `example_id`. Ids already in `per_item/<set>__<mode>.jsonl` are never generated again.
- A partial last line left by a kill is truncated before the run continues.
- A duplicated `example_id` in a per-item file is a hard error.
- An item that raised a pipeline error is recorded as an `error` row and is not retried. This keeps every id at exactly one row.
- The version guards run again on every start. A resumed run therefore refuses if any frozen version changed in between, so a run never mixes states.
- Do not edit code, configs, indexes, adapters or eval sets while a run is in progress or paused (D-048). If a change is needed: stop, document it, and the lead refreezes. The run then starts from a fresh output directory.

## Outputs and privacy
- `per_item/` holds the full QAResponses, which include answer and evidence text. It is gitignored (D-015, D-021).
- Committed artifacts hold only metrics and IDs: `*.meta.json`, and later `artifacts/evaluation/e4/metrics.json` from `score_e4.py`.
- Each per-item row carries `eval_split`, `case_type`, `label_provenance`, `question_provenance` and `reviewer`. Nothing is labelled `human_reviewed`.

## After the run
```
TORCH_DISABLE_NATIVE_JIT=1 HF_HUB_OFFLINE=1 .venv/bin/python scripts/evaluation/score_e4.py \
    --runs artifacts/evaluation/e4/runs --out artifacts/evaluation/e4/metrics.json
```
Scoring uses the pinned NLI checkpoint on the GPU. Run it only after the generation run has finished.
