"""Answerability: build synthetic pairs, train the lexical LR baseline and the Keras BiGRU, compare.

  .venv/bin/python scripts/training/train_answerability.py --stage pairs
  .venv/bin/python scripts/training/train_answerability.py --stage baseline
  .venv/bin/python scripts/training/train_answerability.py --stage keras        # TensorFlow; own process (D-014)
  .venv/bin/python scripts/training/train_answerability.py --stage compare

Thresholds are max-F1 on VALIDATION pairs (D-018); test pairs are scored once with the frozen threshold.
Headline metrics exclude own-answer positives (evaluator condition C4).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from medquad_qa.contracts.answerability import AnswerabilityPair
from medquad_qa.training.answerability_pairs import PAIR_RULE_VERSION, build_pairs, read_pairs, write_pairs
from medquad_qa.training.answerability_train import pair_stats, train_lexical_baseline

SPLITS = ("train", "validation", "test")
BLOCKS = ("val", "val_excl_own_answer", "test", "test_excl_own_answer")
KEYS = ("roc_auc", "pr_auc", "brier", "ece_10bin")
OUT = Path("artifacts/models/answerability")


def _load_pairs() -> dict[str, list[AnswerabilityPair]]:
    return {s: read_pairs(OUT / "pairs" / f"pairs_{s}.jsonl") for s in SPLITS}


def stage_pairs() -> None:
    from medquad_qa.data.corpus import load_split  # verifies sha256 against data/manifests/exports_manifest.json

    stats: dict[str, Any] = {"pair_rule_version": PAIR_RULE_VERSION, "inputs": {}, "splits": {}}
    for split in SPLITS:
        loaded = load_split(split)
        drops: dict[str, int] = {}
        pairs = build_pairs(loaded.records, split, stats=drops)  # type: ignore[arg-type]
        write_pairs(
            pairs,
            OUT / "pairs" / f"pairs_{split}.jsonl",
            OUT / "pairs" / f"pairs_manifest_{split}.jsonl",
            split_version=loaded.split_version,
        )
        stats["inputs"][split] = {
            "sha256": loaded.sha256,
            "corpus_version": loaded.corpus_version,
            "split_version": loaded.split_version,
        }
        stats["splits"][split] = {**pair_stats(pairs), **drops}
    (OUT / "pairs" / "pairs_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats["splits"], indent=2))


def _write(result: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    brief = {s: {k: result[s][k] for k in KEYS} for s in BLOCKS if s in result}
    head = {"model_version": result["model_version"], "threshold": result["threshold"]["threshold"]}
    print(json.dumps({**head, **brief}, indent=2))


def stage_compare() -> None:
    from medquad_qa.training.classifier_metrics import plot_reliability

    runs = {n: json.loads((OUT / n / "metrics.json").read_text()) for n in ("lexical_lr", "keras_bigru")}
    for block in ("val", "test_excl_own_answer"):
        plot_reliability(
            {n: r[block]["reliability_bins"] for n, r in runs.items() if block in r},
            OUT / f"reliability_{block}.png",
            f"Answerability reliability ({block}; synthetic pairs)",
        )
    table = {
        n: {
            "model_version": r["model_version"],
            "threshold": r["threshold"]["threshold"],
            **{f"{s}.{k}": r[s][k] for s in BLOCKS if s in r for k in KEYS},
            **{f"{s}.f1@thr": r[s]["at_threshold"]["f1"] for s in BLOCKS if s in r},
        }
        for n, r in runs.items()
    }
    (OUT / "comparison.json").write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(table, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pairs", "baseline", "keras", "compare"], required=True)
    args = ap.parse_args()
    if args.stage == "pairs":
        stage_pairs()
    elif args.stage == "baseline":
        _write(train_lexical_baseline(_load_pairs(), OUT / "lexical_lr"), OUT / "lexical_lr" / "metrics.json")
    elif args.stage == "keras":
        from medquad_qa.training.keras_answerability import train_keras

        _write(train_keras(_load_pairs(), OUT / "keras_bigru"), OUT / "keras_bigru" / "metrics.json")
    else:
        stage_compare()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
