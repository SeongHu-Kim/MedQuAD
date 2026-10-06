# Security, safety and leakage findings

Owner of this register: **lead** (transferred from evaluation-safety-engineer by D-029). The harness blocks subagents from writing report files, so the evaluator reports each finding by message and the lead transcribes it here. Owners fix their own files, the evaluator retests, and the lead arbitrates.

| ID | Severity | Status | Title | Owner |
|---|---|---|---|---|
| F-001 | medium | open | Cross-split leakage missed by data leakage report | data-steward |

## F-001: Cross-split leakage missed by data leakage report

- **Reported:** 2026-10-06 by evaluation-safety-engineer (pre-freeze E1 check).
- **Severity:** medium. If unfixed, it inflates the Track B (fine-tune generalization) and answerability test metrics. It blocks the E1 sign-off and the split freeze.
- **Reproduction:** `.venv/bin/python scripts/evaluation/verify_splits.py --out artifacts/evaluation/split_verification/report_prefreeze.json` exits 1. The script independently re-derives everything from `medquad.csv`, and the report holds IDs and counts only.
- **Inputs:** split `split-20261006-c759a1668f89`, corpus `medquad-1.0.0-34d97c16127f`.
- **Passed:** reconciliation (16,336 + 76 = 16,412); the record_id formula on every record; raw text preservation; manifest, corpus and export agreement; group purity.
- **Failed** (data-steward's `leakage_report.json` reported passed=true):
  1. 8 topics cross splits because hyphen, apostrophe and comma variants are not folded. Example: beta-ketothiolase deficiency is in test for GHR but in train for GARD.
  2. 10 normalized questions cross splits, for the same reason.
  3. 4 identical disease-specific answers cross splits: the Noonan syndrome 1–6 families, GM1 gangliosidosis types and Osteopetrosis subtypes. The "boilerplate = shared by ≥3 topics" rule classed them as boilerplate, so they never formed union edges.
  4. 33 content near-duplicate pairs cross splits; 32 of them come from the item-3 families.
- **Impact:** 35 split groups (train 18, validation 10, test 7).
- **Affected files:** `src/medquad_qa/data/` (topic fold key, boilerplate rule, leakage checks), `data/manifests/*`, `data/processed/*`.
- **Recommended fix:**
  - (a) Fold keys with NFKC, casefold, and every non-alphanumeric character to a space.
  - (b) Count boilerplate by distinct topic family, or exclude an answer from boilerplate when it contains its own topic name.
  - (c) Add a near-dup check that catches content duplicates across differing topics.
  - (d) Regenerate the split and refreeze.
- **Retest:** pending (same script).
