# Retrieval and grounded RAG

Owner: retrieval-engineer. Code: `src/medquad_qa/retrieval/`, `src/medquad_qa/rag/`. Tests: `tests/retrieval/`, `tests/rag/`.

Status: implemented and tested on **synthetic fixtures** only. No MedQuAD retrieval or QA numbers exist yet; any
future numbers will be added with the command and artifact that produced them.

## Retrievers

All retrievers implement `contracts.Retriever.retrieve(query, top_k) -> list[RetrievalHit]`: ranks 1..n, sorted by
descending score, `[]` on no match, `RetrieverUnavailableError` when the index/backend is missing. Scores are
retriever-specific and uncalibrated; they are never medical confidence.

| Name | What | Needs |
|---|---|---|
| `bm25:answer`, `bm25:qa` | bm25s (Lucene variant, k1=1.5, b=0.75) over chunks | numpy, bm25s (CPU only) |
| `dense:answer`, `dense:qa` | `BAAI/bge-small-en-v1.5` @ `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` (MIT), cosine, Qdrant | torch, sentence-transformers, qdrant-client |
| `hybrid_rrf:*` | RRF (k=60) over BM25 top-50 and dense top-50 record lists | both |
| `hybrid_rrf+ce:*` | hybrid then `cross-encoder/ms-marco-MiniLM-L-6-v2` @ `233902d2…` (Apache-2.0) rerank of top 20 | optional variant |

- **Tokenizer** (`tok-v1`): NFKC + lowercase (index side only), keeps numbers incl. decimals, negations
  (`not`, `no`, `without`, `n't`→`not`) and aspect words (symptoms, treatment, causes). Small closed stopword list.
  Snowball stemming via PyStemmer when installed; negations and numbers are never stemmed.
- **Chunking** (`chunk-v1`): ≤200 words with 40-word overlap; `chunk_id = <record_id>#c<i>`; char offsets into
  `MedicalRecord.answer`; evidence text is always an exact answer substring.
- **Aggregation**: MaxP (record score = best chunk). Optional `--collapse-duplicates` keeps one record per
  `duplicate_group_id`.
- **Index text modes**: `answer` indexes answer chunks; `question_answer` (`*:qa`) prefixes the record's question.
  Evidence returned is always the answer chunk.

> **Disclosure:** `*:qa` indexes contain the original MedQuAD questions. Querying them with original questions is
> **exact-match lookup**, not semantic generalization. Benchmarks must use paraphrased queries and report the two
> modes separately.

- **Lexical fallback**: `MEDQUAD_RETRIEVER=bm25` imports neither torch nor Qdrant. In hybrid mode, if Qdrant/the
  dense index is unavailable, results are BM25-only, named `bm25:*`, and carry the warning `lexical_fallback`.

## Index lifecycle

```bash
python -m medquad_qa.retrieval build   [--kind bm25|dense|all] [--mode answer|question_answer|all]
python -m medquad_qa.retrieval verify  [--kind ...]      # manifest vs files/collection vs loaded corpus; exit 1 on drift
python -m medquad_qa.retrieval rebuild [--kind ...] [--mode ...]
python -m medquad_qa.retrieval delete-local [--kind ...] [--mode ...] [--index-version V] [--purge-manifest]
python -m medquad_qa.retrieval list
python -m medquad_qa.retrieval query --retriever hybrid --mode answer --top-k 5 "question text"
python -m medquad_qa.retrieval run   --retriever hybrid --mode answer --top-k 20 --queries q.jsonl --out run.jsonl
```

- `index_version = <kind>-<answer|qa>-<sha12>` where the hash covers kind, mode, every index-defining parameter
  (chunker, tokenizer, k1/b or embedding model+revision+prefix), the ordered (chunk_id, text) fingerprint and the
  `corpus_version`. Identical inputs give the identical version.
- Manifests: `artifacts/indexes/manifests/<index_version>.json` (git-tracked: IDs, hashes, parameters, environment;
  no corpus text). `manifests/active.json` maps `bm25:answer` etc. to the version retrievers load.
- Data: `artifacts/indexes/bm25/<version>/` (gitignored); Qdrant collection named `<version>` (embedded:
  `artifacts/indexes/qdrant_local/` or `MEDQUAD_QDRANT_PATH`; server: `MEDQUAD_QDRANT_URL`).
- Downloads happen only in `build` (restricted to `*.json`, `*.txt`, `model.safetensors`, `1_Pooling/*`, README).
  Query/serve paths use the local HF cache only; a missing model makes the dense retriever unavailable.
- `run` output (`medquad-retrieval-run-v1`, IDs and scores only):
  `{example_id, retriever, index_version, corpus_version, top_k, latency_ms, warnings, hits:[{record_id, chunk_id, rank, score}]}`.

## RAG pipeline

`medquad_qa.rag.factory.build_pipeline(settings=None, *, observer=None)` → `RagPipeline` (QAPipeline +
ReadinessReporter). Explicit langchain-core LCEL chain:

- RAG modes (`rag`, `finetuned_rag`): `safety_rules → retrieval → answerability_gate → prompt_build → generation → citation_validation`
- Closed-book (`base`, `finetuned`): `safety_rules → prompt_build → generation`

Abstention order (first match wins):

1. Personalized-advice rules (all modes; D-023) → `personalized_medical_advice`. Emergencies get an emergency message.
2. No retrieved hits → `no_relevant_evidence`.
3. Answerability gate rejects the evidence → `no_relevant_evidence`.
4. No evidence fits the input budget → `insufficient_evidence`.
5. Model emits `INSUFFICIENT_EVIDENCE` → `insufficient_evidence`.
6. No valid citation: only invalid ones → `invalid_citations`; none at all → `missing_citations`.

- **Prompts** (`PROMPT_VERSION = rag-v1+<sha8>`): evidence in `<evidence id="E#">` blocks with topic/source lines; the
  system prompt says evidence is untrusted data, requires `[E#]` citations and the sentinel when evidence is
  insufficient.
- **Sanitisation** (`san-v1`): in evidence, topic/source and the question, evidence/question/system tags, chat special
  tokens (`<|…|>`, `[INST]`, `<<SYS>>`), `[E#]` look-alikes, `mq-` record-ID look-alikes, the sentinel and
  control/bidi characters are neutralised. Numbers, units, negations and punctuation are untouched.
- **Citations** (D-010): `[E#]` → `[mq-…]`; a literal `[mq-…]` is kept only if supplied; anything else is stripped
  and listed in `invalid_citation_ids`. Invariant: citations ⊆ supplied ⊆ retrieved.
- **Evidence budget** (D-022): evidence is fitted to the generator's `max_input_tokens` (3072) by keeping the longest
  rank-order prefix of whole hits (warning `evidence_truncated`). Exact counts via `generator.count_tokens` when
  available, otherwise a conservative estimate (3 chars/token).
- **Answerability gate**: trained predictor when available; otherwise a lexical content-coverage heuristic whose
  threshold is fitted (max-F1) on validation pairs only: `python -m medquad_qa.rag.gate_fit --pairs <validation pairs>`.
  Without a fitted file the heuristic only rejects zero-overlap evidence (`heuristic-unfitted`).
  Predictor errors fall back to the heuristic with warning `answerability_fallback`.

## Limitations

- Record-level citations only; MedQuAD's Kaggle export has no URLs (`source_url` stays null).
- The safety rules are regular expressions: they will miss some personal-advice phrasings and may refuse some general
  ones. They are a scope policy, not a clinical classifier.
- Sanitisation prevents forging prompt structure; it cannot guarantee that a model ignores natural-language
  instructions inside evidence.
