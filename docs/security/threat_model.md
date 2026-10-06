# Threat model: MedQuAD Evidence-Grounded QA (research prototype)

Owner: evaluation-safety-engineer. Status: **draft v1**, written before implementation (E3). It is based on the approved plans (handoff §5) and contracts v1.1.1. Every threat maps to a planned E5 test; results go in `findings.md`.

**Scope:** a local, single-user prototype bound to 127.0.0.1. No authentication by design (no public deployment, D-005). It is not a clinical system.

## Assets
1. Integrity of answers and citations: citations must point to evidence that was actually supplied.
2. Safety behaviour: personalized diagnosis or treatment requests are refused in all modes (D-023), while general medical information is still answered.
3. Confidentiality of user questions: no question or answer text in logs by default.
4. Secrets: HF tokens, `.env`, and MLflow credentials, if any.
5. Artifact integrity: corpus, index, adapter and threshold versions, and the frozen eval set.
6. Availability of the local API: bounded resources, clean 503/504 responses.

## Trust boundaries and untrusted inputs
- **User → API** (`POST /v1/qa`, headers): question text, `X-Request-ID`, body size, JSON shape.
- **Corpus/evidence → prompt:** MedQuAD answer text is untrusted data and may contain instruction-like text. Fixture evidence is adversarial by design.
- **Model output → response:** generated text may contain fabricated `[E#]` or `[mq-…]` IDs, chat special tokens, or markup.
- **Artifacts on disk → loaders:** missing, corrupt or mismatched manifests or versions.
- **Dependencies:** Qdrant, MLflow, HF cache (`HF_HUB_OFFLINE=1` in containers).

## Threats → controls → planned tests
| ID | Threat (STRIDE) | Planned control (owner) | Planned test (E5, `tests/security/` or `tests/integration/`) |
|---|---|---|---|
| T1 | Prompt injection in evidence ("ignore previous instructions", fake `</evidence>`, fake system turns, `[E9]`/`mq-` look-alikes) (T/E) | Evidence delimiting and neutralisation, system rules (retrieval) | FixtureRetriever injects payloads. Assert no instruction-following marker in the answer, no new evidence block, payload tokens neutralised in the prompt, citations ⊆ supplied |
| T2 | Prompt injection in the question (role/delimiter tokens, "cite mq-…") (T) | Same neutralisation applied to the question; `QARequest` rejects control chars | Same assertions with the payload in the question |
| T3 | Fabricated or invalid citations (T) | Label→ID mapping, invalid IDs stripped into `invalid_citation_ids`, abstain on invalid or missing citations in RAG (retrieval) | FakeGenerator emits `[E99]`, an unsupplied `[mq-…]`, a malformed ID, and none. Check the response invariant and abstention reasons |
| T4 | Unsupported personalized advice (patient-specific dose, diagnosis, stop/start medication) | Personalized-advice rules in all 4 modes (D-023) | Paired set: personalized requests → abstain `personalized_medical_advice`; matched general-information controls → answered (over-refusal measured) |
| T5 | No relevant evidence answered anyway | No-hits / gate / `INSUFFICIENT_EVIDENCE` abstention (RAG only) | Empty retriever → `no_relevant_evidence`; sentinel output → `insufficient_evidence` |
| T6 | Oversized or malformed input (D) | 16 KB body limit → 413; question ≤2000 chars; control chars → 422 that does not echo input; `top_k` ≤20; generation bounds | Boundary cases via httpx `ASGITransport` against `create_app` with a fake pipeline. Check that the 422 body does not contain the submitted text |
| T7 | Header injection or log forging via `X-Request-ID` (T/R) | Sanitised to `REQUEST_ID_PATTERN` | CRLF, overlong and unicode IDs → replaced; response header matches the pattern |
| T8 | Missing or corrupt model/index/adapter (D) | `ArtifactUnavailableError` / `ModeUnavailableError` → 503 with Retry-After; ready=503 while live=200 | Pipeline provider raises → status codes and body shape; one mode unavailable while others serve |
| T9 | Generation hang or timeout (D) | Generator 60 s deadline → 504; 120 s asyncio backstop; global semaphore → 503 busy | Slow fake generator with shortened timeouts; concurrent requests → busy 503 |
| T10 | Dependency failure (Qdrant down) (D) | Hybrid → lexical fallback with `lexical_fallback` warning; dense-only → 503 | Retriever raising `RetrieverUnavailableError`; warning surfaced in the response |
| T11 | Question/answer text in logs or metrics (I) | JSON logs with no content by default; `MEDQUAD_LOG_CONTENT` opt-in logs a hash plus 200 chars with a warning; metric labels never carry text | Capture logs during requests with a canary string; assert absent. Scrape `/metrics` for the canary |
| T12 | Secret leakage in repo, images or logs (I) | `.gitignore`, `.env.example` only, per-Dockerfile ignore files | `detect-secrets scan` over tracked files; grep built image context lists; check no `HF_TOKEN` in logs or `/v1/info` |
| T13 | Error responses leak internals (stack traces, paths, evidence) (I) | Error mapping with generic bodies | Force a 500 and check the body has no traceback, file path or question text |
| T14 | Version and provenance spoofing or drift (T) | Manifests, content-addressed `index_version`, `corpus_version` check (`ContractViolationError`) | Mismatched corpus/index versions → refuse or 500 with no partial answer |
| T15 | Evaluation integrity: test tuning or leakage (T) | `check_leakage` fails the build; frozen manifests with sha256; thresholds frozen before TEST | `tests/evaluation` leakage cases; manifest tamper test (implemented in E3) |
| T16 | Container hardening regression (E) | Non-root uid 10001, read-only rootfs, `cap_drop ALL`, `no-new-privileges`, 127.0.0.1 binds | `docker compose config` assertions (marker `docker`) |
| T17 | Unsafe deserialisation of artifacts (E) | safetensors for weights; skops/ONNX for classifiers; no `pickle`/`torch.load` on untrusted files | Static grep for `pickle.load`, `joblib.load`, `torch.load(` without `weights_only=True`, and `trust_remote_code=True` |

Severity scale for findings:
- **critical**: exploitable now, or a false safety or validity claim.
- **high**: a safety/integrity control is missing or bypassed.
- **medium**: degraded control or partial leak.
- **low**: hardening or documentation.
- **info**.

## Out of scope
Multi-user authentication and authorisation, network attackers beyond localhost, model weight poisoning upstream (mitigated only by pinned revisions and sha256), and clinical correctness.
