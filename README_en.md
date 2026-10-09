한국어 요약: [README.md](README.md)

# MedQuAD Evidence-Grounded Medical Question Answering — Project Summary

> **Not clinically validated and not medical advice.** This project is a non-clinical research prototype. It does not imply publication, real-world service or employment experience.
>
> **Code generation:** The code in this project was generated with Claude Opus 5.5 (Medium) (the owner's statement). A Claude Code agent team (lead + 5 roles) did the work, and every step went ahead only after the user's approval. The post-closure GUI work (D-069 to D-071) was written by the lead directly.
>
> All numbers come from the evaluation results committed to this repository (the E6 report and the decision records D-xxx).

## 1. At a glance

- **What was built:** a system that answers medical-information questions while citing its evidence, and declines to answer when there is no evidence or when the question asks for personal medical advice.
- **What was compared:** a base model, retrieval-based (RAG) and fine-tuning models, compared under the same conditions as 6 configurations built from 4 modes and 2 adapters (v1, v2c).
- **Best-balanced mode:** **rag** (base model + retrieval). Its accuracy metric (reference coverage) showed no significant difference from the base model, but its answers can be verified and it declines most questions without evidence.
- **Key finding:** fine-tuning did not raise the accuracy metric. The model trained on the citation format had the side effect of copying the evidence verbatim.
- **Remaining problems:** malicious instructions hidden in evidence documents (injection attacks) were not stopped, and personal-advice and crisis handling (F-009) is also unresolved under the rules. So no mode is fit for real use.

## Try the GUI (Tailscale)

**Why is the Tailscale app needed?**
This GUI is not open to the public internet. Because safety problems remain unresolved (see section 9), it is offered only through **Tailscale**, a private network that only invited people can join. The Tailscale app connects your device securely to this private network. The models and data stay on the operator's computer; you only use the interface in your browser.

The GUI offers 4 modes (base, rag, finetuned, finetuned_rag) and uses the v1 adapter (v2c is not offered). It runs the current version (safety-v4, evidence filter ef-v1), so its behaviour may differ from the figures in 7.1 and 7.2.

**Please follow these rules before use**
- This is a research prototype; it is not clinically validated and not medical advice.
- Do not enter real personal health information (your own or anyone else's symptoms, diagnoses, medications, etc.).
- Do not act on any answer. Consult a medical professional about health problems.
- At most **5 people** are invited during the test period, and access ends when the operator ends it. It is available only while the operator's computer is on.
- If you are actually in distress, contact real help, not this system: in Korea, 109 (suicide prevention line, 24 hours) or 119 for emergencies.

**How to connect**
1. **Request an invitation:** email rickseonghukim@gmail.com saying that you would like to use the GUI described in the https://github.com/SeongHu-Kim/MedQuAD repository and need a private Tailscale invite link. No name or other information is needed.
2. **Receive the invite link:** the operator replies with a Tailscale invite link. The link is single-use, for one person, and must be used within 30 days. Do not forward it to anyone else.
3. **Accept the invitation:** open the invite link, sign in with the account you want to use (Google, Microsoft, GitHub, etc.) and accept.
4. **Install the app:** install the Tailscale app for your device (Windows, macOS, Linux, Android, iPhone) from https://tailscale.com/download.
5. **Sign in to the app:** open the app and sign in with the account you used to accept the invitation.
6. **Connect to the network:** in the app's network (tailnet) list, select **seonghu-kim.github** (SeongHu-Kim@github) and check that the status is **Connected**.
7. **Open the GUI:** open https://promaxgb10-bddf.tail386012.ts.net in your browser. The first connection may take a little time.
8. **Use it:** choose a mode on the left, then click an "예시 질문" (example question) or type a question **in English**, and press "질문하기 (Ask)".
9. **Finish:** when you are done, turn off the connection in the app.

If you have a problem, contact rickseonghukim@gmail.com (do not include personal health information).

## 2. Project overview

This system uses the MedQuAD data (medical questions and answers from US NIH sources). When a user asks a general medical-information question, it finds relevant records, answers and cites the sources. When evidence is insufficient or the question asks for personal medical advice (dosage, stopping treatment, etc.), it declines to answer (abstains).

All main comparisons were run as **pre-declared evaluations**. That is, the metrics, pass criteria and statistical methods were fixed before any results were seen, so that the criteria could not be changed to fit the results. Analyses added after the results were seen are marked separately as "post hoc".

## 3. The 6 configurations compared (4 modes, 2 adapters)

| Mode | Retrieval (RAG) | Adapter | Description |
|---|---|---|---|
| **base** | No | None | The base model answers from memory only |
| **rag** | Yes | None | Reads the retrieved evidence and answers with citations |
| **finetuned v1** | No | v1 (in use) | A model LoRA-trained on MedQuAD answers answers from memory |
| **finetuned_rag v1** | Yes | v1 | The v1 model + retrieval |
| **finetuned v2c** | No | v2c (not promoted) | A model also trained on the citation format answers from memory |
| **finetuned_rag v2c** | Yes | v2c | The v2c model + retrieval |

## 4. Pipeline and tools used

The project went through the following 7 stages.

> **[Data] → [Retrieval index] → [Training] → [RAG pipeline] → [Service] → [Evaluation · Security] → [MLOps · Reproducibility]**

| Stage | Tools used | Main work |
|---|---|---|
| **1. Data** | Python 3.12, MedQuAD CSV (Kaggle, local) | Medical text cleaning and quality audit. Group-level split to reduce overlap between training and test data (absence not proven; 16,336 → 13,014 / 1,665 / 1,657) |
| **2. Retrieval index** | bm25s (BM25), BAAI/bge-small-en-v1.5 (embeddings), **Qdrant** (vector DB), RRF hybrid, MiniLM cross-encoder | Compared 8 retrieval methods and chose the best |
| **3. Training** | **PyTorch** 2.14.1 (CUDA 13), Hugging Face **Transformers**, **PEFT (LoRA)**, **TensorFlow/Keras** (BiGRU), logistic regression | LLM fine-tuning (v1, v2c) and training an "is this question answerable?" classifier |
| **4. RAG pipeline** | **LangChain-core**, Qwen/Qwen3-4B-Instruct-2507 (generation model), safety rules safety-v4, evidence filter ef-v1 | Safety rules → retrieval → evidence filter → answerability check → answer generation → citation validation |
| **5. Service** | **FastAPI**, **Streamlit** (demo interface), **Docker Compose** (local only) | API and interface, run in containers |
| **6. Evaluation · Security** | DeBERTa NLI (automatic scoring), pytest (972 offline tests and 150 security tests passing; as of D-071, 942 at the D-066 reproduction check) | Pre-declared evaluation, statistical tests (Holm correction, bootstrap, McNemar, Wilson intervals), injection-attack and personal-advice security tests |
| **7. MLOps · Reproducibility** | **MLflow** (experiment tracking), **Prometheus** (monitoring), Git · GitHub (private), ruff · mypy, requirements.lock | Pinned versions, checksums, decision records (docs/decisions.md, from D-001), clean-environment reproduction check. GitHub Actions CI: ARM64 passing (x86 had 2 failures from a test-fixture problem on the first run; fixed but not yet re-run. The Keras classifier is not checked in CI, D-079) |

- **Hardware:** NVIDIA GB10 (CUDA 13, 128 GB unified memory, 121 GiB usable). No paid cloud was used.
- **Development:** a Claude Code agent team worked in separate roles (data, retrieval, model, service, evaluation · security + lead). Every commit and GPU run was approved by the user.
- **Post-closure work (D-068 to D-074):** the operator's local GUI testing, an example-question panel with a behaviour check (not an accuracy judgement), a Korean GUI with mode-selection buttons, and private test access through Tailscale. Evaluation results and findings did not change.

## 5. Evaluation by stage (success or not)

Each stage was compared with **completion criteria fixed in advance**. "Execution" means whether it was carried out as planned; "outcome" means whether it reached its goal.

✅ Success · ⚠️ Partial success · ❌ Below criteria

> **[Data ✅] → [Retrieval index ✅] → [Training ⚠️] → [RAG pipeline ⚠️] → [Service ⚠️] → [Evaluation ✅ · Security ❌] → [MLOps · Reproducibility ✅]**

| Stage | Verdict | Evidence | Remaining limits |
|---|---|---|---|
| **1. Data** | ✅ | The reasons for all 76 excluded rows are recorded. An independent rebuild was fully identical. The first leakage check failed (F-001) but passed its retest after the fix | Data version unverified (licence CC BY-SA 4.0 as stated on Kaggle) |
| **2. Retrieval index** | ✅ | dense:qa was best of 8 methods (R@5 0.950) | The index contains the original questions, so the figure is optimistic (0.92 on AI-written questions) |
| **3. Training** | ⚠️ | Execution succeeded (v1 790/790, v2c 607/607 steps completed). But no accuracy gain. v1 cannot cite, and v2c copies the evidence | The logistic regression (0.826) is served instead of the better classifier (BiGRU 0.961) |
| **4. RAG pipeline** | ⚠️ | Citation validity 1.000. Fictional-condition and out-of-scope questions 100% declined. The new safety rules met every criterion on unseen items, but those figures are for reference only and the finding is unresolved under the rules (D-063) | About 19% of answerable questions wrongly declined (Track A 57/300; 0.183 on the 60 Track C items). Injection attacks unresolved |
| **5. Service** | ⚠️ | API and interface tests passed. Run and rollback drill passed in the GPU environment (pre-fix version) | The real interface check (S3) was not run as an evaluation (only post-closure personal and invited testing, D-068 · D-073). API response time not measured. Public deployment blocked |
| **6a. Evaluation** | ✅ | Criteria fixed before results were seen. Private test items registered by hash and run only once. A mistake (a statistical criterion changed after the fact) was disclosed and reverted by the user's decision (D-055) | Evaluation items were written by AI and not reviewed by a person. The lead's viewing of some evaluation-item information during development was disclosed (D-053) |
| **6b. Security** | ❌ | The injection-attack fix missed its criterion (at most 1 of 40) (rag 13/40, v2c 19/40) | 3 high-severity problems unresolved |
| **7. MLOps · Reproducibility** | ✅ | After a reinstall in a clean environment, lint and 942 offline tests passed. Every decision is recorded and evidence is kept with checksums | The v2c training was not logged to MLflow. Disk use over budget |

**Overall:** data, retrieval, evaluation and reproducibility met their criteria. Training, RAG and service were executed successfully but their outcomes are limited. Security missed its pre-set criteria.

## 6. Evaluation metrics explained

| Metric | What it shows |
|---|---|
| **Answer rate** | The share of questions actually answered rather than declined |
| **Reference coverage** | How much of the MedQuAD reference answer an answer contains. **The main metric standing in for accuracy**; declined questions score 0 |
| **Unsupported-claim rate** | The share of an answer's claims not supported by the reference answer or the supplied evidence (lower is better) |
| **Citation validity** | Whether a cited source is really part of the supplied evidence (no fake citations) |
| **Citation support** | Whether a cited source really supports the sentence (judged automatically by an NLI model) |
| **Copy rate** | Whether an answer copies evidence sentences verbatim (high means copying, not summarising) |
| **Abstention recall** | The share of questions that should not be answered (fictional conditions, out of scope, insufficient evidence) that were correctly declined |
| **Over-refusal rate** | The share of answerable questions wrongly declined (lower is better) |
| **Personal-advice refusal / crisis message** | Whether requests such as personal dosage are declined, and crisis statements get a help message |
| **Injection leak** | Whether an answer follows instructions hidden in the evidence documents (lower is better) |
| **Harmful-instruction following** | Whether an answer to an injected item requesting a medical action followed that action (judged by an AI agent) |
| **Memorisation probe** | Whether the model reproduces training-data answers from memory |

> **Note:** every accuracy metric is an **automatic agreement** with the MedQuAD reference answers. Whether answers are medically correct (clinical judgement) was not measured. In the tables, `[a, b]` is a 95% confidence interval and "—" is a value that was not reported. "No difference" means that no statistically significant difference was detected; it does not prove the two values are identical.

## 7. Results

> 7.1, 7.2 and 7.4 are results of the pre-fix version (E4, 1c84b60: safety-v2, no evidence filter). The current version (a6b7d3e: safety-v4, evidence filter ef-v1) was measured only in the retest in 7.3, and the regression check on ordinary questions was not run (D-065).

### 7.1 300 answerable questions (Track A)

| Metric | base | rag | finetuned v1 | finetuned_rag v1 | finetuned v2c | finetuned_rag v2c |
|---|---|---|---|---|---|---|
| Answer rate | 300/300 | 243/300 | 300/300 | **7/300** | — | 225/300 |
| Reference coverage | 0.384 | 0.379 | 0.354 (−0.030 vs base) | 0.023 | −0.001 vs v1 | 0.498 (**+0.120** vs rag) |
| Citation validity | n/a | 1.000 | n/a | 1.000 (n = 7) | n/a | 1.000 |
| Citation support | n/a | 0.857 | n/a | 1.000 (n = 7) | n/a | 0.947 (rag 0.863 on items both answered, +0.084) |
| Copy rate | n/a | 0.221 | n/a | — | n/a | **0.9996** |

- **rag vs base:** coverage difference −0.005 [−0.051, +0.042]. **No significant difference was detected** (by the 95% interval, any difference is roughly in the range −5 to +4 percentage points).
- **finetuned v1 vs base:** −0.030 [−0.075, +0.015]. **No training effect was confirmed.**
- **finetuned_rag v1:** it did not learn the citation format, so 266 of 300 were declined as "no citation".
- **finetuned_rag v2c:** the scores rose, but because it copied the evidence almost verbatim (copy rate 0.9996). These cannot be considered better answers.
- **Note (post hoc analysis, not confirmatory):** on questions both modes answered, rag's unsupported-claim rate was 45 percentage points lower than base's (243 items).

### 7.2 Difficult questions (Track C, retrieval modes)

| Metric | rag | finetuned_rag v1 | finetuned_rag v2c |
|---|---|---|---|
| Out-of-scope / fictional-condition questions declined | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 |
| Questions with insufficient evidence declined | 0.84 | 0.98 (49/50) | 0.88 |
| Over-refusal (60 answerable) | 0.183 | — (0 of 40 general questions answered) | 0.117 |
| Injection leak (20) | 8/20 | 1/20 (almost never answers) | **19/20** |
| Answered when the evidence conflicts | 7/20 | — | 19/20 |

- The modes that answer from memory (base, finetuned) do not read evidence, so they are not exposed to injection attacks. Instead, they can make up plausible answers even about fictional conditions (not measured).
- Personal-advice refusal was the same **30/40** in every mode, because all modes use the same safety rules (pre-fix rules, on Track C).
- 20 of the 60 over-refusal items contain injected evidence.
- finetuned_rag v1's refusals are mostly due to missing citations (missing_citations).

### 7.3 Retest after the fix (unseen items, run once)

| Item | Result [95% interval] | Criterion |
|---|---|---|
| Personal-advice refusal (all modes) | 60/60 [0.940, 1.000] | 55/60 or more |
| Crisis message (all modes) | 30/30 [0.886, 1.000] | 29/30 or more |
| General questions wrongly declined | 0/60 [0.000, 0.060] | 1/60 or fewer |
| Injection leak: rag | **13/40** [0.201, 0.480], harmful-instruction following 7/10 | 1/40 or fewer |
| Injection leak: finetuned_rag v2c | **19/40** [0.329, 0.625], harmful-instruction following 10/11 | 1/40 or fewer |
| Injection leak: finetuned_rag v1 | 0/40 (but meaningless, since it answered none) | 1/40 or fewer |

- **New safety rules (safety-v4):** they passed everything on the unseen items. But because they had earlier missed an internal criterion by one item, they were recorded as **unresolved** under the pre-set rules. The figures above are for reference only.
- **Evidence filter:** leaks were 0/41 on the development items, but new forms of attack were not stopped.
- The harmful-instruction-following judgements were made by an AI agent against pre-set criteria and were not reviewed by a person.

### 7.4 Other results

- **Retrieval performance:** R@1 0.830, R@5 0.950, MRR 0.884. The index contains the original questions, so these figures are optimistic (R@5 0.92 on AI-written questions).
- **Answerability classifier (AUROC):** the logistic regression used in the service scored 0.826. But on "same topic, different question type" it scored 0.536, close to random. The Keras BiGRU was better at 0.961 but was not used in the service.
- **Memorisation probe (100 items):** the difference between finetuned v1 and base was −0.027 [−0.100, +0.045]. No sign of memorised training data was found.
- **API response time:** not measured.

## 8. Assessment by mode and the best-balanced mode

| Mode | Assessment |
|---|---|
| base | Always answers, but has no evidence or citations, so it cannot be verified. It can also make up fictional conditions. |
| **rag** | Its accuracy metric shows no significant difference from base. But **every citation is valid**, and it declines most questions without evidence. Its weaknesses are over-refusal (about 19% of the 300 Track A questions) and injection attacks. |
| finetuned v1 | No significant difference from base. Fine-tuning did not raise the accuracy metric. |
| finetuned_rag v1 | It cannot cite, so it answered only 7 of 300. In practice it does not work. |
| finetuned v2c | The same level as v1, neither worse nor better. |
| finetuned_rag v2c | It solved the citation problem but copies the evidence verbatim and is the most vulnerable to injection attacks (19/20, 19/40). It was decided not to use it. |

### Best-balanced mode: **rag** (base model + retrieval)

1. **It can be verified.** Every citation is part of the supplied evidence (validity 1.000), and about 86% support their sentence (NLI automatic judgement). Users can check the sources themselves.
2. **It declines most questions without evidence.** It declined 100% of fictional-condition and out-of-scope questions and 84% of difficult questions with insufficient evidence.
3. **No accuracy loss was detected.** The reference coverage difference was −0.005 [−0.051, +0.042].
4. **It is more robust.** It copies less than v2c (0.221 vs 0.9996) and leaks less under injection (13/40 vs 19/40).
5. **It is simple.** It needs no extra training (adapter).

> "Best-balanced" means relatively so among the 6 configurations, not that it is a validated best configuration. rag does not stop injection attacks either (13/40), so **no mode is fit for real use.**

## 9. Safety and security status

| Problem | Severity | Status |
|---|---|---|
| **F-009** Personal advice and crisis handling | high | Passed on unseen items, but unresolved because the pre-set criterion was missed |
| **F-010** Following instructions in the evidence | high | Fix failed (rag 13/40) |
| **F-011** v2c copies hidden text | high | Fix failed (19/40) |

All three problems were **recorded and accepted as known limitations** by the user (D-066). Public deployment and demos are blocked (D-052). The only exceptions are the operator's own local testing (D-068) and private Tailscale testing by at most 5 invited people (D-073, D-074). Nothing observed in that testing is evaluation evidence.

## 10. Limitations

- Every evaluation item and label was written from templates or by AI and was not reviewed by a person. The harmful-instruction-following judgements were also made by an AI agent.
- Accuracy is only an automatic agreement with the MedQuAD reference answers and was not clinically validated. Citation support is also an automatic NLI judgement.
- During development the lead agent viewed some information about protected evaluation items; this was recorded and disclosed (D-053).
- API response time and the four-mode regression check after the fix were not run.
- The MedQuAD data licence is CC BY-SA 4.0 as stated on the Kaggle page (checked by the operator on 2026-10-09; not compared with the original release). The data version could not be confirmed.

## 11. Conclusion

**Achievements:** the retrieval-based answers (rag) kept an **accuracy metric with no significant difference** from the base model while producing **verifiable answers**. Every citation was part of the supplied evidence (validity 1.000), and about 86% supported their sentence (NLI automatic judgement). It also did not make up answers to fictional-condition or out-of-scope questions, declining **100%** of them, and declined 84% of difficult questions with insufficient evidence. The new safety rules (safety-v4) correctly handled **60/60** personal-advice requests and **30/30** crisis statements on unseen items. These cannot be compared directly with the old rules (30/40 on Track C) because the item sets differ, and they are reference figures only because the finding is unresolved under the rules. In the evaluation, criteria were fixed before results were seen and the private test items were run only once, so every result was recorded traceably.

Of these, **source citation and verification** and **declining questions without evidence** are improvements the base model could not provide. The base model answers from memory only, so it has no sources to cite and cannot judge whether evidence exists (fictional-condition items were measured only in the retrieval modes). Because rag finds and reads the evidence itself, it made both possible without a detected loss in the accuracy metric. The new safety rules, by contrast, run before the answer in every mode, so they are an improvement that applies equally to the base model and not an achievement of rag alone.

**Limitations:** rag did not raise answer accuracy (reference coverage) itself, and it wrongly declined about 19% of answerable questions (Track A 57/300). Fine-tuning the 4B model gave no measurable benefit, and the model trained on the citation format (v2c) showed the side effect of copying the evidence verbatim. Despite good results, the new safety rules missed a pre-set internal criterion by one item and remain unresolved under the rules. Above all, **injection attacks through instructions hidden in evidence documents were not solved** (rag 13/40). For this reason no mode is fit for real use.

The core achievement of this project is that criteria were fixed before results were seen, and **both good and bad results were recorded honestly**.

## 12. References

1. **Dataset (Kaggle):** MedQuAD: Medical Question-Answer for AI Research.
   https://www.kaggle.com/datasets/pythonafroz/medquad-medical-question-answer-for-ai-research
2. **Cited paper (cited as the dataset page asks):** Ben Abacha, A., & Demner-Fushman, D. (2019). A Question-Entailment Approach to Question Answering. *BMC Bioinformatics*, 20(1), 511. https://doi.org/10.1186/s12859-019-3119-4
3. **Data licence:** CC BY-SA 4.0 (as stated on the Kaggle page).
   https://creativecommons.org/licenses/by-sa/4.0/

## 13. Developer quick start

```bash
make venv && make install        # CUDA 13 torch (aarch64) + extras; see requirements.lock
make lint && make test           # offline: no GPU, models or Docker needed
python -m medquad_qa.data build  # requires a local medquad.csv (not redistributed)
make up PROFILE=gpu              # local stack on 127.0.0.1; no public demos (D-052; exceptions D-068, D-073)
```

A fresh clone with a new environment built from `requirements.lock` passed `make lint` and 942 offline tests (D-066); the offline suite has 972 tests as of D-071. Data builds, indexes, training and evaluations need the local CSV, the GPU and git-ignored artifacts; see [docs/data/reproduction.md](docs/data/reproduction.md), [docs/retrieval/README.md](docs/retrieval/README.md), [docs/models/](docs/models/), [docs/evaluation/e4_runbook.md](docs/evaluation/e4_runbook.md) and [docs/operations/runbook.md](docs/operations/runbook.md).
