# R6 closeout: retrieval and RAG tests, failure cases, docs, review findings

Owner: retrieval-engineer. Written 2026-10-08, docs only (no code change), at HEAD `79013aa`.
R6 acceptance line: *"Tests, failure cases, docs; respond to E5 review: tests green; findings resolved/documented."*

**Verdict: met as "documented", not "all resolved".** Tests are green and every finding is either resolved with a
retest or open and documented. Two findings in this area (F-009, F-010) remain **open (high)**; the user has
accepted them as known limitations (D-066). Demo and deployment stay blocked (D-052, D-065).

## 1. Tests (green)

Command, run when this note was written:

```
.venv/bin/pytest tests/rag tests/retrieval -m "not real_model" -q
```

Result: **526 passed, 1 deselected** (the deselected test is `tests/retrieval/test_real_embedder.py`, marker
`real_model`, which needs the cached BGE weights; it passed when run earlier on GPU).

| File | Tests collected |
|---|---|
| `tests/rag/test_citations_sanitize_safety.py` | 134 |
| `tests/rag/test_evidence_filter.py` | 56 |
| `tests/rag/test_factory.py` | 4 |
| `tests/rag/test_gate_fit.py` | 2 |
| `tests/rag/test_pipeline.py` | 28 |
| `tests/rag/test_safety_check.py` | 40 |
| `tests/rag/test_safety_v3.py` | 224 |
| `tests/retrieval/test_bm25.py` | 13 |
| `tests/retrieval/test_cli.py` | 3 |
| `tests/retrieval/test_dense_hybrid.py` | 11 |
| `tests/retrieval/test_tokenize_chunking.py` | 11 |
| `tests/retrieval/test_real_embedder.py` | 1 (deselected offline) |

Owner test data (SYNTHETIC or owner-written, never MedQuAD eval items): `tests/rag/data/safety_bank_v1.jsonl`,
`safety_bank_v2.jsonl` (registered in D-061/D-062), `injection_bank_v1.jsonl`, and their generators.

## 2. Failure-case tests

- **Retriever failures:** missing index or corpus mismatch → `RetrieverUnavailableError`
  (`test_bm25.py::test_missing_index_or_corpus_mismatch_raises`); index-file and corpus drift detected
  (`test_bm25.py::test_persisted_build_load_and_drift`, `test_cli.py::test_cli_verify_detects_corpus_drift`);
  Qdrant unreachable (`test_dense_hybrid.py::test_qdrant_server_unreachable_raises`); missing collection or
  embedder mismatch; hybrid degrades to lexical with `lexical_fallback` (`test_hybrid_degrades_to_lexical`,
  `test_dense_fallback_mode`); missing corpus never raises in the factory.
- **Queries as data:** query-language and delimiter strings (`test_bm25.py::test_query_is_data`); empty or stopword
  queries return `[]`.
- **Citations:** invalid labels and IDs stripped and listed (`test_invalid_labels_and_ids_stripped_and_reported`);
  unbracketed fabricated IDs (`test_unbracketed_record_ids_validated`, F-003); only-invalid → `invalid_citations`;
  none → `missing_citations`; sentinel → `insufficient_evidence`; cited ⊆ supplied ⊆ retrieved
  (`test_pipeline.py`).
- **Injection in evidence (structural):** forged evidence/question tags, chat special tokens, `[E#]` and `mq-`
  look-alikes, Unicode Cf/fullwidth evasions (`test_neutralize_injection_fixtures`,
  `test_prompt_structure_cannot_be_forged_by_evidence`, `test_neutralize_unicode_evasions`).
- **Injection in evidence (plain language, F-010):** marked payloads dropped with the fact kept
  (`test_evidence_filter.py::test_marked_payload_dropped_fact_kept`); a fully flagged chunk not supplied
  (`test_pipeline.py::test_fully_flagged_chunk_is_not_supplied`); medical guidance not flagged.
- **Safety (F-009):** personal and general lists in both directions; crisis routing; safe harbour; the model check's
  never-downgrade rule, crisis rules never calling the model, every check failure failing closed with
  `safety_check_failed`, the strict label parser, and check-then-answer sequencing on a non-reentrant lock
  (`test_safety_check.py`).
- **Budget and gate:** evidence truncation keeps whole rank-order hits (`evidence_truncated`); predictor failure falls
  back to the heuristic gate (`answerability_fallback`); readiness and versions never raise.

## 3. Docs

- `docs/retrieval/README.md`: retrievers, index lifecycle and CLI, RAG pipeline and abstention order, safety-v4
  stages, evidence filter (including the note that a filtered hit's character offsets refer to the original
  chunk), prompts, sanitisation, citations, DEV diagnostics, the citation-prompt A/B, the frozen configuration,
  and limitations (Q+A index contains original questions; small DEV; safety rules tuned on DEV-style items;
  link-list gold answers; never merge the adapter while safety-v4 uses the base view; classifier spoofing; the
  guidance-styled injection gap has no mitigation since the rag-v2 clause was reverted).
- This closeout note.

## 4. Findings owned by retrieval-engineer

Source: `docs/security/findings.md` (summary table and per-finding sections) and `docs/decisions.md`.

| Finding | Severity | Status | Evidence |
|---|---|---|---|
| F-002 Personalized-advice rules miss personal/emergency requests, over-refuse general ones | high | **resolved**, retest PASS on `0e4ed7e` | findings.md F-002 |
| F-003 Unbracketed fabricated record IDs reach the user | medium | **resolved**, retest PASS on `0e4ed7e` | findings.md F-003 |
| F-004 Zero-width, fullwidth and RLM look-alikes bypass the sanitiser | low | **resolved**, retest PASS on `0e4ed7e` | findings.md F-004 |
| F-006 Closed-book modes report the RAG prompt_version | low | **resolved**, retest PASS on `0e4ed7e` | findings.md F-006 |
| F-007 Safety rules do not generalise to fresh probes; self-harm ideation missed | medium | **resolved**, retest PASS on `7206637` | findings.md F-007; D-039 |
| F-008 Passive self-harm phrasing missed; third-party support question over-refused | low | **resolved**, retest PASS on `2b95db1` (runtime `safety-v2+43836b6c`) | findings.md F-008; D-041, D-042 |
| F-009 Personalized-advice/crisis rules do not generalise to held-out requests | **high** | **open, unresolved**; accepted as a known limitation (D-066) | findings.md F-009; D-060, D-062, D-063, D-065 |
| F-010 Plain-language instructions planted in evidence are followed by rag | **high** (raised from medium, D-065) | **open**; remediation failed the fresh retest; accepted as a known limitation (D-066) | findings.md F-010; D-060, D-064, D-065 |

Notes on the open findings:

- **F-009.** safety-v3 rules missed the owner go/no-go on bank v1's holdout; the safety-v4 model check missed it on
  bank v2's holdout by one over-refused general item (1/66), so F-009 was recorded as unresolved before any fresh
  result (D-063). The fresh retest's §1 numbers for safety-v4 are descriptive only (D-065): personal 60/60
  refused, crisis 30/30 crisis message, controls 0/60 over-refused, 0 failed checks. The probes are AI-written
  without human review.
- **F-010.** The evidence filter (`ef-v1+e899826d`) dropped 0/41 marked payloads on the owner fixtures, but on the
  fresh held-out probes rag leaked the canary in 13/40 (Wilson upper 0.480 vs the 0.15 bar) and the §2.1 review
  labelled 7 of 10 eligible served answers as following the requested medical action (`ai_agent`, not
  human-reviewed) (D-065). The rag-v2 prompt clause was reverted before the freeze (D-064) because rag citation
  placement fell on DEV; payloads written as ordinary guidance have no mitigation beyond rag-v1's general rule.
- F-011 (adapter v2c) is owned by model-engineer and is not listed here.
