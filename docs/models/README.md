# docs/models (owner: model-engineer)

| Document | Content |
|---|---|
| `generator_card.md` | Model card: Qwen3-4B base + LoRA adapter (identity, use, settings, limitations) |
| `inference.md` | Generator API, error behaviour, environment variables, host-specific fixes, benchmark |
| `training.md` | Closed-book LoRA SFT pipeline and the M4 run results |
| `answerability_card.md` | Model card: synthetic answerability pairs, LR baseline (serving gate) and Keras BiGRU |

Evidence (all under `artifacts/models/`):
- `manifests/base_model.json`: pinned revision, licence and per-file sha256 of the base model.
- `bench/generator_base_{cuda,cpu_smoke}.json`: load time, latency, tokens/s and memory.
- `runs/<run_id>/`: `run_manifest.json`, `metrics.jsonl`, `loss_curve.png`, `sft_build_*.json`, `sft_record_ids.jsonl`,
  `reload_check.json`.
- `adapters/CURRENT`: the promoted adapter run id.
- `answerability/`: pair manifests and stats, `{lexical_lr,keras_bigru}/metrics.json`, IDs-only score files,
  `comparison.json` and reliability plots.

Tests:
- Offline: `.venv/bin/pytest tests/models tests/training -m "not real_model"`. Uses tiny random-init Qwen3, synthetic
  records, and Keras in a subprocess.
- Real weights: `.venv/bin/pytest tests/models/test_models_real_generator.py -m real_model`.
