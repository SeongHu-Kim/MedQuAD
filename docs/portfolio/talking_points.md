# Interview talking points

Owner: lead. Short, evidence-backed points on design choices and limitations. Every number links to its source. These describe a portfolio research prototype; they make no claim of clinical validation, publication or employment experience.

## Design choices

- **Group-aware splits before anything else.** The export has no document IDs, so splits group by content and topic, with leakage checks. The first verification failed (F-001), the rules were fixed, and the retest passed before the split was frozen (D-031, D-034; E6 §2).
- **Evaluation frozen before experiments.** Sets, metrics and Holm families were declared before TEST output. One post-hoc enlargement of a family was caught, disclosed and reverted (D-055).
- **Citations are checked, not trusted.** The pipeline only accepts citations to evidence it actually supplied (cited ⊆ supplied ⊆ retrieved), and that invariant held in every run (E6).
- **Abstain rather than guess.** An answerability gate plus a generator sentinel. Measuring them separately showed the served gate was near chance on same-topic negatives (AUROC 0.536) and the sentinel did most of the work (E6 §5).
- **Safety rules run in every mode,** including closed-book, so mode comparisons are like for like (D-023).

## Results worth discussing

- **Retrieval:** a dense index over question+answer text was best (R@5 0.950), but it contains the original questions, so the number is optimistic; on AI-written questions it was 0.92 (E6 §4).
- **RAG vs base:** no reference-coverage difference was detected (−0.005 [−0.051, +0.042]); RAG abstained more. Its lower unsupported-claim rate is exploratory (E6 §6).
- **Fine-tuning:** closed-book LoRA showed no gain. Citation-formatted fine-tuning fixed citation output (225/300 vs 7/300 answered) but learned to copy single records (copy rate 0.9996), so its metric gains were not presented as better answers (D-058).

## Negative results, owned

- **Injection remediation failed on fresh payloads.** An evidence filter reached 0/41 leaks on development fixtures and 13/40 on fresh held-out items (D-065). The lesson: development fixtures written by the same team do not cover an attacker's styles.
- **Safety remediation missed its own bar by one item** (1/66 over-refusal against a bar of 0/66), so F-009 stayed unresolved even though the fresh numbers looked good (D-063, D-065). Holding to a pre-declared rule mattered more than a better-looking result.
- **A prompt change was reverted** because it lowered citation coverage on DEV with no measured benefit (D-064).

## Process

- A five-role agent team with one owner per path; every step gated by the user; every number traced to a committed artifact.
- Held-out data was protected by hash registration, guard tests and disclosure of the two coincidental collisions (D-059, D-061, D-062).

## Limitations to state up front

- All evaluation items and labels are synthetic and not human-reviewed; the harm review was done by an AI agent.
- Three high findings are open; demo and deployment are blocked.
- API latency was not measured; the regression check on the remediated pipeline was not run.
