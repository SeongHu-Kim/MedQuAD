# Split design and leakage controls

Owner: data-steward. Parameters live in `configs/data/build.yaml`. Resulting counts are in
[`audit_report.md`](audit_report.md) §4–6 and in `data/manifests/split_manifest.meta.json`.

## Why groups

The export has no document IDs. One MedQuAD source document usually yields several rows (one per
question type) under one `focus_area`. The same condition also appears under several sources, and some
answers are copied or templated across conditions. Splitting by row would leak the same document, or
near-identical text, between train and test. All splitting therefore happens at the level of
`split_group_id`.

## Groups (union-find, `grouping.py`)

| edge | duplicate_group | split_group |
|---|---|---|
| Same normalized answer (excluding boilerplate) | yes | yes |
| Same (normalized question, normalized answer) | yes | yes |
| Plain near-duplicate answer: word 5-shingles, Jaccard ≥ `near_dup_jaccard` | yes | yes |
| Same folded topic key (across sources) | – | yes |
| Same folded topic key after removing parenthetical text ("X" vs "X (abbrev)") | – | yes |
| Same folded question | – | yes |
| Template near-duplicate: Jaccard ≥ `near_dup_jaccard` after masking each record's own topic name | – | yes |

**Near-duplicate candidates** are pairs that share at least `min_shared_rare_shingles` shingles with document frequency ≤
`rare_shingle_max_df`. Their exact Jaccard is then computed over the full shingle sets. Identical shingle sets
are collapsed first and linked with J = 1.

**Folded keys** (`normalize.match_key`): NFKD with combining marks removed, casefold, and every run of characters
outside `[0-9a-z]` becomes one space. "Coffin-Lowry syndrome", "Coffin Lowry Syndrome", "Graves' Disease" and
"Graves disease" therefore each fold to one key. Keys are used only for matching.

**Boilerplate answers** are normalized answers shared by at least `boilerplate_min_topics` distinct topics
whose folded keys do **not** all start with the same word. They create **no** edges, because otherwise one
generic sentence (e.g. the autosomal-recessive inheritance paragraph) would chain unrelated conditions into a
single group. They carry the `boilerplate_answer` flag and may appear in several splits. An answer shared only
within one topic family ("Noonan syndrome 1…6", "GM1 gangliosidosis type 1…3") is disease-specific content.
It is not boilerplate, so it links the family into one group.

**Invariant:** every duplicate group lies inside one split group.

Group IDs are `dg-` or `sg-` followed by sha256 of the sorted member record IDs. They change only when
membership changes.

## Assignment (`splits.py`)

1. Each split group gets a **dominant source**: its most frequent `source`, with ties broken
   alphabetically. This is the stratum.
2. Within each stratum, groups are sorted by ID and shuffled with `random.Random(f"{seed}|{stratum}")`.
   The seed is `20261006`.
3. Groups are assigned greedily to the split furthest below its row target (0.8 / 0.1 / 0.1).

The split labels are `train`, `validation` and `test`, the same names used by `AnswerabilityPair.split`.

## Leakage checks (`leakage.py`)

**Blocking checks.** The build and `verify` fail unless every count below is 0:
- the same split_group in more than one split
- the same duplicate_group in more than one split
- the same topic key in more than one split
- the same bracket-stripped topic key in more than one split
- the same folded question in more than one split
- the same non-boilerplate normalized answer in more than one split
- a plain near-duplicate pair (J ≥ threshold) that crosses splits
- a template near-duplicate pair (J ≥ threshold) that crosses splits

**Diagnostics.** These are reported in `leakage_report.json` and never silently fixed:
- boilerplate answers that span splits (allowed by design)
- residual pairs with topic-masked Jaccard in [`residual_jaccard_min`, threshold) that cross splits,
  listed by record ID and J
- topics that merge only after their parenthetical text is removed, listed with the splits they touch
  (whether such a merge crosses splits is itself a blocking check)

## Revision history

| split_version | change |
|---|---|
| `split-20261006-c759a1668f89` | First candidate. Rejected by evaluator finding F-001 (`docs/security/findings.md`): the folded keys kept hyphens, apostrophes and commas, the boilerplate rule swallowed answers shared across numbered subtypes, and the rare-shingle df ≤ 10 filter missed content near-duplicates |
| `split-20261006-dd1d7f31e9bc` | Alphanumeric folding, family-aware boilerplate rule, `rare_shingle_max_df` 10 → 25 |
| `split-20261006-2f0fb25ee6d8` (**FROZEN**, evaluator sign-off 2026-10-06) | Evaluator CR3: topics that match after their parenthetical alias is removed ("X" vs "X (abbrev)") now link split groups, and a cross-split merge is a blocking check. Regression tests cover each F-001 class plus CR3: the pair must be linked, and a leak must be detected if forced |

## Evaluator sign-off

The evaluation-safety-engineer reviews the split design before it is declared frozen. The sign-off
outcome and any requested changes are reported to the lead and recorded in `docs/decisions.md`. After
the freeze, changing any grouping parameter changes `split_version`, and downstream artifacts must be
rebuilt.

## What this does not guarantee

- Paraphrased content with low lexical overlap is not detected.
- Different conditions that share pathophysiology can still teach transferable facts across splits.
- Topic names that differ in spelling, beyond case, whitespace and trailing punctuation, are not
  merged. Bracketed aliases are handled; topic families such as "Vitamin A" and "Vitamin C" are
  deliberately not merged.
