"""Grounded QA pipeline as an explicit LangChain (langchain-core LCEL) chain.

RAG modes:    safety_rules -> retrieval -> answerability_gate -> prompt_build -> generation -> citation_validation
Closed-book:  safety_rules -> prompt_build -> generation

Abstention order (first match wins):
  1. personalized-advice rules (all modes; D-023)        -> personalized_medical_advice
  2. no retrieved hits                                    -> no_relevant_evidence
  3. answerability gate says evidence is not adequate     -> no_relevant_evidence
  4. no evidence fits the input budget                    -> insufficient_evidence
  5. model emits the INSUFFICIENT_EVIDENCE sentinel        -> insufficient_evidence
  6. no valid citation: only invalid ones / none at all   -> invalid_citations / missing_citations
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.runnables import Runnable, RunnableLambda

from medquad_qa.contracts import (
    RAG_MODES,
    AbstentionReason,
    ArtifactUnavailableError,
    ChatMessage,
    Citation,
    ComponentStatus,
    ExperimentMode,
    GenerationMeta,
    GenerationParams,
    GenerationResult,
    Generator,
    ModeUnavailableError,
    PipelineObserver,
    PredictorOutput,
    QARequest,
    QAResponse,
    RetrievalHit,
    Retriever,
    RetrieverUnavailableError,
)
from medquad_qa.rag.budget import EVIDENCE_TRUNCATED, fit_evidence, token_counter_for
from medquad_qa.rag.citations import build_citations, validate_citations
from medquad_qa.rag.gate import Gate, HeuristicGate
from medquad_qa.rag.prompts import (
    CLOSED_BOOK_PROMPT_VERSION,
    PROMPT_VERSION,
    EvidenceBlock,
    build_closed_book_messages,
    build_rag_messages,
)
from medquad_qa.rag.safety import SAFETY_RULES_VERSION, check_question

log = logging.getLogger(__name__)

GeneratorProvider = Callable[[str], Generator]

ABSTENTION_MESSAGES: dict[str, str] = {
    "no_relevant_evidence": (
        "I could not find relevant evidence in the MedQuAD corpus to answer this question, so I am not answering it."
    ),
    "insufficient_evidence": (
        "The retrieved evidence does not contain enough information to answer this question reliably."
    ),
    "invalid_citations": (
        "The generated answer cited evidence that was not supplied, so it was withheld. Please rephrase or try again."
    ),
    "missing_citations": "The generated answer did not cite the supplied evidence, so it was withheld.",
}


@dataclass
class _State:
    request: QARequest
    request_id: str
    generator: Generator
    t0: float
    timings: dict[str, float] = field(default_factory=dict)
    done: bool = False
    abstention: AbstentionReason | None = None
    answer: str = ""
    hits: list[RetrievalHit] = field(default_factory=list)
    supplied: list[RetrievalHit] = field(default_factory=list)
    retriever_name: str | None = None
    warnings: list[str] = field(default_factory=list)
    gate_output: PredictorOutput | None = None
    gate_used: str | None = None
    messages: list[ChatMessage] = field(default_factory=list)
    result: GenerationResult | None = None
    citations: list[Citation] = field(default_factory=list)
    invalid_ids: list[str] = field(default_factory=list)

    @property
    def mode(self) -> ExperimentMode:
        return self.request.experiment_mode


def _variant(mode: str) -> str:
    return "finetuned" if mode in ("finetuned", "finetuned_rag") else "base"


class RagPipeline:
    """Implements ``QAPipeline`` and ``ReadinessReporter``. All collaborators are injected (DI constructor)."""

    def __init__(
        self,
        *,
        retriever: Retriever | None,
        generator_provider: GeneratorProvider,
        gate: Gate | None = None,
        observer: PipelineObserver | None = None,
        max_input_tokens: int = 3072,
        component_statuses: list[ComponentStatus] | None = None,
        generator_status: Callable[[str], ComponentStatus] | None = None,
    ) -> None:
        self.retriever = retriever
        self._provider = generator_provider
        self._generators: dict[str, Generator] = {}
        self._generator_errors: dict[str, str] = {}
        self._lock = threading.Lock()
        self.gate = gate or Gate(None, HeuristicGate())
        self.observer = observer
        self.max_input_tokens = max_input_tokens
        self._statuses = list(component_statuses or [])
        self._generator_status = generator_status
        self.prompt_version = PROMPT_VERSION
        self._rag_chain = self._build_rag_chain()
        self._closed_chain = self._build_closed_chain()

    # ------------------------------------------------------------------ generators
    def get_generator(self, variant: str) -> Generator:
        with self._lock:
            gen = self._generators.get(variant)
            if gen is not None:
                return gen
            try:
                gen = self._provider(variant)
            except ArtifactUnavailableError as exc:
                self._generator_errors[variant] = str(exc)
                if variant == "finetuned":
                    raise ModeUnavailableError("finetuned", str(exc)) from exc
                raise
            self._generators[variant] = gen
            self._generator_errors.pop(variant, None)
            return gen

    def preload(self, variants: tuple[str, ...] = ("base", "finetuned")) -> None:
        """Load generators eagerly; failures are recorded for readiness, never raised."""
        for v in variants:
            try:
                self.get_generator(v)
            except ArtifactUnavailableError as exc:
                log.warning("generator %s unavailable: %s", v, type(exc).__name__)
            except Exception as exc:
                self._generator_errors[v] = type(exc).__name__
                log.warning("generator %s failed to load: %s", v, type(exc).__name__)

    # ------------------------------------------------------------------ observer
    def _notify(self, fn_name: str, *args: Any) -> None:
        if self.observer is None:
            return
        try:
            getattr(self.observer, fn_name)(*args)
        except Exception:  # observers must never break answering
            log.debug("observer %s failed", fn_name, exc_info=True)

    # ------------------------------------------------------------------ chain steps
    @staticmethod
    def _timed(key: str, fn: Callable[[_State], None]) -> Runnable[_State, _State]:
        def step(state: _State) -> _State:
            if state.done:
                return state
            t = time.perf_counter()
            try:
                fn(state)
            finally:
                state.timings[key] = round((time.perf_counter() - t) * 1000, 3)
            return state

        return RunnableLambda(step).with_config(run_name=key)

    def _abstain(self, state: _State, reason: AbstentionReason, message: str | None = None) -> None:
        state.done = True
        state.abstention = reason
        state.answer = message or ABSTENTION_MESSAGES.get(reason, "I cannot answer this question.")
        self._notify("on_abstention", state.mode, reason)

    def _step_safety(self, state: _State) -> None:
        decision = check_question(state.request.question)
        if decision.refuse:
            self._abstain(state, "personalized_medical_advice", decision.message)

    def _step_retrieve(self, state: _State) -> None:
        if self.retriever is None:
            raise RetrieverUnavailableError("no retriever configured")
        r: Any = self.retriever
        if hasattr(r, "retrieve_with_info"):
            hits, name, warnings = r.retrieve_with_info(state.request.question, state.request.top_k)
        else:
            hits, name, warnings = r.retrieve(state.request.question, state.request.top_k), r.name, []
        state.hits, state.retriever_name = list(hits), name
        state.warnings.extend(warnings)
        self._notify("on_retrieval", state.mode, state.hits)
        if not state.hits:
            self._abstain(state, "no_relevant_evidence")

    def _step_gate(self, state: _State) -> None:
        output, used, warnings = self.gate.evaluate(state.request.question, [h.evidence_text for h in state.hits])
        state.gate_output, state.gate_used = output, used
        state.warnings.extend(warnings)
        self._notify("on_gate", state.mode, output, used)
        if output is not None and not output.predicted_label:
            self._abstain(state, "no_relevant_evidence")

    def _step_prompt_rag(self, state: _State) -> None:
        question = state.request.question

        def build(hits: Any) -> list[ChatMessage]:
            blocks = [
                EvidenceBlock(label=f"E{i}", text=h.evidence_text, topic=h.topic, source=h.source_name)
                for i, h in enumerate(hits, start=1)
            ]
            return build_rag_messages(question, blocks)

        count, _ = token_counter_for(state.generator)
        limit = int(getattr(state.generator, "max_input_tokens", self.max_input_tokens))
        supplied, messages, truncated = fit_evidence(state.hits, build, count, limit)
        if truncated:
            state.warnings.append(EVIDENCE_TRUNCATED)
        if not supplied:
            self._abstain(state, "insufficient_evidence")
            return
        state.supplied, state.messages = supplied, messages

    def _step_prompt_closed(self, state: _State) -> None:
        state.messages = build_closed_book_messages(state.request.question)

    def _step_generate(self, state: _State) -> None:
        params = state.request.generation or GenerationParams()
        state.result = state.generator.generate(state.messages, params)
        state.answer = state.result.text.strip()

    def _step_citations(self, state: _State) -> None:
        res = validate_citations(state.answer, state.supplied)
        state.invalid_ids = res.invalid_ids
        state.citations = build_citations(res.cited_record_ids, state.supplied)
        self._notify("on_citations", state.mode, state.citations, state.invalid_ids)
        if res.has_sentinel:
            state.citations = []
            self._abstain(state, "insufficient_evidence")
        elif not state.citations:
            self._abstain(state, "invalid_citations" if res.invalid_ids else "missing_citations")
        else:
            state.answer = res.text

    def _build_rag_chain(self) -> Runnable[_State, _State]:
        return (
            self._timed("safety_rules", self._step_safety)
            | self._timed("retrieval", self._step_retrieve)
            | self._timed("answerability_gate", self._step_gate)
            | self._timed("prompt_build", self._step_prompt_rag)
            | self._timed("generation", self._step_generate)
            | self._timed("citation_validation", self._step_citations)
        )

    def _build_closed_chain(self) -> Runnable[_State, _State]:
        return (
            self._timed("safety_rules", self._step_safety)
            | self._timed("prompt_build", self._step_prompt_closed)
            | self._timed("generation", self._step_generate)
        )

    # ------------------------------------------------------------------ QAPipeline
    def answer(self, request: QARequest, request_id: str) -> QAResponse:
        t0 = time.perf_counter()
        mode = request.experiment_mode
        is_rag = mode in RAG_MODES
        if is_rag and self.retriever is None:
            raise ModeUnavailableError(mode, "retriever unavailable")
        generator = self.get_generator(_variant(mode))
        state = _State(request=request, request_id=request_id, generator=generator, t0=t0)
        chain = self._rag_chain if is_rag else self._closed_chain
        state = chain.invoke(state)
        return self._response(state, is_rag)

    def _response(self, s: _State, is_rag: bool) -> QAResponse:
        result = s.result
        generation = None
        if result is not None:
            generation = GenerationMeta(
                params=s.request.generation or GenerationParams(),
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                finish_reason=result.finish_reason,
            )
        if result is not None and is_rag and s.supplied:
            retrieved_ids = [h.record_id for h in s.supplied]
        else:
            retrieved_ids = [h.record_id for h in s.hits]
        citations = [] if s.abstention else s.citations
        r: Any = self.retriever
        return QAResponse(
            request_id=s.request_id,
            experiment_mode=s.mode,
            answer=s.answer,
            abstained=s.abstention is not None,
            abstention_reason=s.abstention,
            citations=citations,
            retrieved_record_ids=retrieved_ids,
            invalid_citation_ids=s.invalid_ids,
            retriever=s.retriever_name if is_rag else None,
            answerability=s.gate_output,
            answerability_gate=s.gate_used if is_rag and s.gate_used else None,
            generation=generation,
            warnings=list(dict.fromkeys(s.warnings)),
            model_version=s.generator.model_version,
            corpus_version=getattr(r, "corpus_version", None) if is_rag else None,
            index_version=(s.hits[0].index_version if s.hits else getattr(r, "index_version", None))
            if is_rag
            else None,
            prompt_version=self.prompt_version if is_rag else CLOSED_BOOK_PROMPT_VERSION,
            latency_ms=round((time.perf_counter() - s.t0) * 1000, 3),
            component_latency_ms=s.timings,
        )

    # ------------------------------------------------------------------ ReadinessReporter
    def readiness(self) -> list[ComponentStatus]:
        try:
            out = [s for s in self._statuses]
            names = {s.name for s in out}
            for variant in ("base", "finetuned"):
                name = f"generator:{variant}"
                if name in names:
                    continue
                gen = self._generators.get(variant)
                if gen is not None:
                    out.append(
                        ComponentStatus(name=name, ok=True, required=variant == "base", version=gen.model_version)
                    )
                else:
                    detail = self._generator_errors.get(variant, "not loaded")
                    if self._generator_status is not None and variant not in self._generator_errors:
                        try:  # file-level check only; never loads weights
                            st = self._generator_status(variant)
                            detail = f"not loaded; {st.detail or ('files ok' if st.ok else 'files missing')}"
                        except Exception:
                            log.debug("generator_status failed", exc_info=True)
                    out.append(ComponentStatus(name=name, ok=False, required=variant == "base", detail=detail))
            if "answerability" not in names:
                out.append(
                    ComponentStatus(
                        name="answerability",
                        ok=True,
                        required=False,
                        degraded=self.gate.mode != "predictor",
                        detail=f"gate={self.gate.mode}",
                        version=self.gate.threshold_version,
                    )
                )
            return out
        except Exception as exc:  # contract: never raises
            return [ComponentStatus(name="pipeline", ok=False, required=True, detail=type(exc).__name__)]

    def close(self) -> None:
        """Release generator references and close the retriever's Qdrant client if any. Idempotent."""
        with self._lock:
            self._generators.clear()
        client = getattr(getattr(self.retriever, "dense", None), "client", None) or getattr(
            self.retriever, "client", None
        )
        if client is not None:
            try:
                client.close()
            except Exception:
                log.debug("qdrant client close failed", exc_info=True)

    def available_modes(self) -> frozenset[ExperimentMode]:
        modes: set[ExperimentMode] = set()
        if "base" in self._generators:
            modes.add("base")
            if self.retriever is not None:
                modes.add("rag")
        if "finetuned" in self._generators:
            modes.add("finetuned")
            if self.retriever is not None:
                modes.add("finetuned_rag")
        return frozenset(modes)

    def versions(self) -> dict[str, str | None]:
        r: Any = self.retriever
        base = self._generators.get("base")
        ft = self._generators.get("finetuned")
        return {
            "model_version": base.model_version if base else None,
            "finetuned_model_version": ft.model_version if ft else None,
            "corpus_version": getattr(r, "corpus_version", None),
            "index_version": getattr(r, "index_version", None),
            "retriever": getattr(r, "name", None),
            "prompt_version": self.prompt_version,
            "closed_book_prompt_version": CLOSED_BOOK_PROMPT_VERSION,
            "safety_rules_version": SAFETY_RULES_VERSION,
            "answerability_threshold_version": self.gate.threshold_version,
        }
