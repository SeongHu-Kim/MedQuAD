# Demo script

> **DO NOT PERFORM while D-052 stands.** Demo, deployment and user-facing serving are blocked: three high-severity findings are open (F-009, F-010, F-011), accepted only as documented limitations (D-066). This script records what a demo *would* show once the user lifts the block. It is not evidence that a demo took place.

Owner: lead. Duration: about 8 minutes. Local machine only (127.0.0.1).

## Before starting (only after the block is lifted)

1. Confirm the user has approved a demo in writing (a decision row).
2. Rebuild the image from the approved commit and run `make gpu-check`, `make up PROFILE=gpu` and `make stack-smoke` (`docs/operations/runbook.md`).
3. Use only invented, general questions. Never use a real person's situation, and never use evaluation or probe items.

## Script

1. **Framing (30 s).** "This is a non-clinical research prototype over the MedQuAD corpus. It is not medical advice and is not clinically validated."
2. **Grounded answer (90 s).** Ask a general question in `rag` mode. Show the answer, the `[mq-…]` citations, the cited evidence snippets, and the version block from `/v1/info`.
3. **Abstention (60 s).** Ask about an invented condition. Show the abstention reason (`no_relevant_evidence` or `insufficient_evidence`).
4. **Personal-advice refusal (60 s).** Ask an invented personal-dosing question. Show the refusal. For a crisis-style statement, show that the crisis message names emergency services. Say plainly that F-009 is unresolved.
5. **Mode comparison (90 s).** Ask one question in `base`, `rag` and `finetuned`, and point to E6 §6 for the measured differences. Do not demonstrate `finetuned_rag` with v1 (it rarely answers; E6 §6.1).
6. **Monitoring (60 s).** Show `/metrics`: request counts, abstention reasons, warnings and component latency labels. Show that logs contain no question or answer text.
7. **Limitations (60 s).** State F-010/F-011: planted instructions in evidence still get through on fresh payloads (13/40 and 19/40 leaked, D-065). Close with the limitations list in the README.

## What not to show or claim

- No claim of clinical validity, accuracy for real patients, or safety for real users.
- No live injection demonstration against the corpus.
- No performance or latency numbers that are not in a committed artifact; API latency was never measured.
