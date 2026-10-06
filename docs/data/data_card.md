# Data card: MedQuAD corpus (`medquad_qa.data`)

Owner: data-steward. All counts are in [`audit_report.md`](audit_report.md), which is rendered from
`data/manifests/*.json` by `python -m medquad_qa.data report`. This card does not repeat numbers, so it
cannot drift from them.

## Source

| item | value |
|---|---|
| File | `medquad.csv` at the repo root (never modified; gitignored) |
| Integrity | sha256 pinned in `configs/data/build.yaml` and checked on every build; recorded in `data/manifests/source_dataset.json` |
| Columns | `question`, `answer`, `source`, `focus_area` (the header is validated exactly) |
| Origin | A Kaggle CSV export of MedQuAD (Ben Abacha & Demner-Fushman, 2019), compiled from NIH websites (GARD, GHR, NIDDK, NINDS, MedlinePlus, NIHSeniorHealth, CancerGov, NHLBI, CDC) |
| Kaggle owner/slug | **Not verifiable** from the file. Stored as `kaggle_dataset_ref = null` |
| Licence | **Unverified.** Local research use only. Do not redistribute the dataset or its text. Tracked files contain IDs and topics only (D-015) |
| Missing metadata | No source URLs, document IDs or original MedQuAD question types. `source_url` and `source_document_id` are always `null` (D-004) |

## Processing

1. **Ingest** (`ingest.py`): stdlib `csv`, UTF-8, every row read verbatim.
2. **Normalize** (`normalize.py`): Unicode NFC, newline unification, per-line whitespace collapse (NBSP
   and tabs included), blank-line collapse. Questions also get `" ?" → "?"`.
   - No lowercasing, and no edits to punctuation, numbers, units, negations or spelling.
   - The originals stay in `question_raw` and `answer_raw`.
   - Folded keys are used only for matching and are never stored as text. They apply accent
     stripping, casefolding and non-alphanumeric → space; see `split_design.md`.
3. **Exclusions** (`quality.py`): the first matching rule wins. Each excluded row is listed in
   `data/manifests/exclusions.jsonl` with its `row_index`.

   | reason | rule |
   |---|---|
   | `empty_answer` | Normalized answer is empty. Kept separate from short valid answers, which stay in the corpus and get the flag `short_answer` |
   | `exact_duplicate_row` | All four raw columns are identical to an earlier row. The copy is mapped to its canonical record (`canonical_record_id`, `Provenance.duplicate_row_indices`) |
   | `non_informative_answer` | Either `question_only` (the whole answer is a single question sentence) or `heading_only` (at most `heading_max_words` words and no line ends a sentence) |

4. **Record IDs** (`ids.py`): `"mq-" + sha256("medquad-rid-v1" US source US focus_area_raw US question_raw US answer_raw)[:16]`.
   - The ID is content-derived, does not depend on row order, and is checked for uniqueness on every build.
   - `content_hash` is sha256 of the normalized question + US + the normalized answer.
5. **Question types** (`question_types.py`, `configs/data/question_types.yaml`): ordered regex
   templates. They are labelled `rule_version` and are **not** the original MedQuAD qtype.
6. **Quality flags**: flags describe the text and never change it.

   | flag | meaning |
   |---|---|
   | `boilerplate_answer` | The normalized answer is shared by at least `boilerplate_min_topics` distinct topics that are not one topic family (they do not all start with the same word) |
   | `malformed_question` | The question has no final `?`, or contains `??` (the export template "Who is at risk for X? ?") |
   | `repeated_bullets` | Heuristic: at least 2 bullet items recur verbatim later in the answer (export artefact) |
   | `answer_starts_with_question` | The answer opens by restating a question ("What causes X? …") |
   | `short_answer` | At most `short_answer_max_words` words (valid but short) |
   | `missing_topic` | `focus_area` is empty |
   | `collapsed_exact_duplicates` | Exact duplicate rows were folded into this record |

7. **Grouping, splits and leakage checks**: see [`split_design.md`](split_design.md).

## Outputs

| path | tracked | content |
|---|---|---|
| `data/processed/corpus.jsonl` | no | One `MedicalRecord` per line, including text |
| `data/processed/exports/records_{train,validation,test}.jsonl` | no | The same records, partitioned by split |
| `data/manifests/source_dataset.json` | yes | Source file hash, size, rows and licence note |
| `data/manifests/corpus_manifest.json` | yes | `corpus_version`, corpus sha256, counts and reconciliation |
| `data/manifests/exclusions.jsonl` | yes | `row_index`, reason, detail, source, topic and canonical record |
| `data/manifests/split_manifest.jsonl` | yes | `record_id`, split, group IDs, source, topic, question_type and flags |
| `data/manifests/split_manifest.meta.json` | yes | `split_version`, seed, ratios, per-split counts and question-type rule table |
| `data/manifests/leakage_report.json` | yes | Blocking checks plus diagnostics (residual pairs and bracket merges) |
| `data/manifests/exports_manifest.json` | yes | Export paths, record counts and sha256 |
| `data/manifests/audit.json` | yes | All audit counts and distributions |

Versions:
- `corpus_version = medquad-<semver>-<sha256(corpus.jsonl)[:12]>`
- `split_version = split-<seed>-<sha256(split_manifest.jsonl)[:12]>`

Outputs contain no timestamps or absolute paths, so a rebuild is byte-identical
(`python -m medquad_qa.data verify`).

## Status

Split frozen on 2026-10-06 after evaluator sign-off (E1, F-001 retest PASS):
`split-20261006-2f0fb25ee6d8` on corpus `medquad-1.0.0-86e384302357`. See `split_design.md` for the
revision history. Any later change to the data rules produces a new split_version and needs a new sign-off.

## Known limitations

- Grouping is derived from content and topic because document metadata is absent. It is weaker than
  document-level grouping: leakage is reduced, **not proven absent**.
- Near-duplicate detection needs shared rare shingles. Very short answers are matched only exactly, and
  some pairs that overlap only on common template text may be missed. Residual similar pairs across
  splits (Jaccard 0.5–0.8) are listed by ID, not hidden.
- Topic families (numbered subtypes such as "Dystonia 1" / "Dystonia 11" or "Spinocerebellar ataxia N")
  are **not** grouped, because a suffix-stripping rule would also merge unrelated topics such as
  "Vitamin A" / "Vitamin C". The evaluator's crude family rule finds cross-split families. Their answers
  are distinct (no cross-split content near-duplicates), so this is disclosed and not counted as leakage.
  Answers copied verbatim across a family *are* linked (family-aware boilerplate rule).
- A duplicate group can contain near-identical templated answers about different conditions. Downstream
  collapse should also key on topic.
- Boilerplate answers appear in several splits by design.
- Answer text keeps the export's artefacts (repeated lists, headings run into text). They are flagged,
  not repaired, so that medical content is never altered.
- Medical content dates from the original crawl and may be outdated. This is a research prototype, not
  clinical guidance.
