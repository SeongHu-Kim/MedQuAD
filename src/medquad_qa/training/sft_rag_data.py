"""Mixed SFT examples for adapter v2 (D-051): closed-book + RAG-formatted, [E#]-cited targets.

Everything is built from ONE split (train for training, validation for validation loss), never the production
index:
- Evidence pool: the split's non-boilerplate records, chunked with retrieval's chunker (<=200 words, 40 overlap).
- Distractors: a dense index built in-process over that pool with an injected embedder. Real runs use
  BAAI/bge-small-en-v1.5 in question_answer mode, the same as serving's dense:qa (D-037).
- rag_answerable: the gold record's first chunk is placed at a random position among k blocks. The others are top
  dense hits from different split groups. The target is the gold chunk's own sentences, each followed by its
  [E#] label (extractive, single source), cut at a sentence boundary within ``max_target_tokens``, then EOS.
- rag_insufficient: k top hits from different split groups AND topics, none textually equal to the gold chunk.
  The target is exactly ``INSUFFICIENT_EVIDENCE``.
- closed_book: the v1 builder (``sft_data.SFTBuilder``) unchanged.
Each question gets at most one format (seeded hash order). Every RAG target is checked with the pipeline's
``validate_citations``.

sft-mix-v2b (option b'; enabled when ``MixPlan.same_topic_weights`` is set, otherwise v2 is unchanged):
- On-topic = D-031 ``topic_key`` equal and non-None. Same-topic candidates are records of the question's own split
  group and topic with a DIFFERENT question_type, passing guards G1-G7 (see ``_Ctx.candidates``), each represented by
  its best-scoring passing chunk.
- rag_answerable adds m same-topic blocks, m ~ ``same_topic_weights`` over 0..4 (serving distribution), capped by
  availability and k-1. rag_insufficient shows T = m'+1 same-topic blocks from the same weights (capped by
  availability and k), so on-topic totals match across classes. A further ``n_sentinel_info`` sentinels are
  information-type questions with off-topic evidence only.
- Same-topic sentinels are drawn only from allowed question types with >=1 guarded candidate; questions that do
  not qualify are skipped (counted) and fall through to closed-book in seeded order.
- All distractors are ordered by dense score; length drops remove off-group blocks before same-topic ones.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from medquad_qa.contracts.qa import RetrievalHit
from medquad_qa.contracts.records import MedicalRecord
from medquad_qa.data.normalize import topic_key
from medquad_qa.models.answerability import LexicalFeaturizer, content, tokens
from medquad_qa.rag.citations import validate_citations
from medquad_qa.rag.prompts import SENTINEL, EvidenceBlock, build_rag_messages
from medquad_qa.rag.sanitize import neutralize
from medquad_qa.retrieval.corpus import Chunk, chunk_record, index_text
from medquad_qa.training.sft_data import (
    EXCLUDED_FLAGS,
    IGNORE_INDEX,
    MessageBuilder,
    SFTBuilder,
    SFTExample,
    default_closed_book_messages,
    file_sha256,
    read_records,
)

MIX_VERSION = "sft-mix-v2"
GUARDS = (
    "G1_same_qtype",
    "G2_gold_dup_group",
    "G3_excluded_flags",
    "G4_text_equal_gold",
    "G5_gold_jaccard",
    "G6_coverage_ge_gold",
    "G7_sentinel_qtype_pair",
)
Format = Literal["closed_book", "rag_answerable", "rag_insufficient"]
_SENT = re.compile(r"(?<=[.!?])\s+")


class DocEmbedder(Protocol):
    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    def encode_queries(self, texts: list[str]) -> np.ndarray: ...


@dataclass
class MixPlan:
    n_rag_answerable: int
    n_rag_insufficient: int
    n_closed_book: int
    k: int = 5
    max_target_tokens: int = 256
    candidates: int = 50
    seed: int = 20261006
    mix_version: str = MIX_VERSION
    # ---- sft-mix-v2b (all unset -> v2 behaviour)
    same_topic_weights: list[float] | None = None  # P(m = 0..4) extra same-topic blocks (serving distribution)
    sentinel_selection: str = "top_score"  # top_score | random (same-topic block choice for sentinels)
    sentinel_excluded_qtypes: dict[str, list[str]] = field(default_factory=dict)  # G7: question qtype -> block qtypes
    sentinel_disallowed_qtypes: list[str] = field(default_factory=list)  # never drawn as same-topic sentinels
    n_sentinel_info: int = 0  # of n_rag_insufficient: info-type questions with off-topic evidence only
    info_qtype: str = "information"
    jaccard_max: float = 0.5  # G5: drop candidates with content-token Jaccard >= this vs the gold chunk
    # False (sft-mix-v2c, option D): every sentinel shows off-topic evidence only; no candidate-based skips
    sentinel_same_topic: bool = True

    @property
    def sentinel_kind(self) -> str:
        """Row ``sentinel_kind`` of the non-information sentinels."""
        return "sentinel_same_topic" if self.sentinel_same_topic else "sentinel_offtopic"

    @property
    def v2b(self) -> bool:
        return self.same_topic_weights is not None

    def as_dict(self) -> dict[str, Any]:
        d = dict(vars(self))
        if not self.v2b:  # keep the committed sft-mix-v2 report byte-identical
            for key in list(d):
                if key not in _V2_PLAN_KEYS:
                    d.pop(key)
        elif self.sentinel_same_topic:  # keep the sft-mix-v2b report byte-identical
            d.pop("sentinel_same_topic")
        return d


_V2_PLAN_KEYS = (
    "n_rag_answerable",
    "n_rag_insufficient",
    "n_closed_book",
    "k",
    "max_target_tokens",
    "candidates",
    "seed",
)


@dataclass
class MixReport:
    mix_version: str
    plan: dict[str, Any]
    counts: Counter[str] = field(default_factory=Counter)
    skipped: Counter[str] = field(default_factory=Counter)
    distractors_dropped_for_length: int = 0
    gold_position: Counter[int] = field(default_factory=Counter)
    rag_prompt_tokens: list[int] = field(default_factory=list)
    closed_book_report: dict[str, Any] | None = None
    v2b: dict[str, Any] | None = None  # v2b-only diagnostics (guards, histograms, cue check); see v2b_summary

    def to_dict(self) -> dict[str, Any]:
        p = sorted(self.rag_prompt_tokens)
        out = {
            "mix_version": self.mix_version,
            "plan": self.plan,
            "counts": dict(self.counts),
            "skipped": dict(self.skipped),
            "distractors_dropped_for_length": self.distractors_dropped_for_length,
            "gold_position": {f"E{k}": v for k, v in sorted(self.gold_position.items())},
            "rag_prompt_tokens": {
                "n": len(p),
                "median": p[len(p) // 2] if p else None,
                "p90": p[int(0.9 * (len(p) - 1))] if p else None,
                "max": p[-1] if p else None,
            },
            "closed_book": self.closed_book_report,
        }
        if self.v2b is not None:
            out["v2b"] = self.v2b
        return out


def _order_key(seed: int, record_id: str) -> str:
    return hashlib.sha256(f"{seed}:{record_id}".encode()).hexdigest()


def _norm(text: str) -> str:
    return " ".join(text.split())


def cited_target(chunk_text: str, label: str, fits: Any) -> tuple[str, int, int]:
    """Sentences (and bullet lines) of the chunk, each followed by ' [label]'; longest prefix that ``fits``.

    Returns (text, sentences_used, sentences_total)."""
    pieces: list[tuple[str, bool]] = []  # (sentence, starts_new_line)
    for line in neutralize(chunk_text).splitlines():
        sents = [s.strip() for s in _SENT.split(line) if s.strip()]
        pieces.extend((s, i == 0) for i, s in enumerate(sents))
    out, used = "", 0
    for sent, new_line in pieces:
        candidate = (out + ("\n" if new_line else " ") if out else "") + f"{sent} [{label}]"
        if not fits(candidate):
            break
        out, used = candidate, used + 1
    return out, used, len(pieces)


class MixedSFTBuilder:
    def __init__(
        self,
        tokenizer: Any,
        embedder: DocEmbedder,
        *,
        max_seq_len: int = 2048,
        closed_book_builder: MessageBuilder = default_closed_book_messages,
        closed_max_seq_len: int = 1024,
    ) -> None:
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.max_seq_len = max_seq_len
        # closed-book examples keep v1's 1024 limit so they are built exactly as in v1 (same truncation)
        self.closed = SFTBuilder(tokenizer, max_seq_len=closed_max_seq_len, message_builder=closed_book_builder)
        self.eot_ids = self.closed.eot_ids

    def _ids(self, text: str) -> list[int]:
        return [int(t) for t in self.tokenizer.encode(text, add_special_tokens=False)]

    def _prompt_ids(self, question: str, blocks: list[EvidenceBlock]) -> list[int]:
        msgs = [m.model_dump() for m in build_rag_messages(question, blocks)]
        ids = self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=False)
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        return [int(t) for t in ids]

    def build(
        self, records: Sequence[MedicalRecord], plan: MixPlan
    ) -> tuple[list[SFTExample], MixReport, list[dict[str, Any]]]:
        """Returns (examples, report, id_rows). ``id_rows`` are IDs-only manifest rows (one per example)."""
        if plan.v2b:
            return self._build_v2b(records, plan)
        report = MixReport(mix_version=plan.mix_version, plan=plan.as_dict())
        rng = random.Random(f"{plan.seed}:mix")  # noqa: S311 - reproducible sampling
        pool = [r for r in records if not EXCLUDED_FLAGS.intersection(r.quality_flags) and r.answer.strip()]
        ordered = sorted(pool, key=lambda r: _order_key(plan.seed, r.record_id))
        n_a, n_i = plan.n_rag_answerable, plan.n_rag_insufficient
        rag_a, rag_i = ordered[:n_a], ordered[n_a : n_a + n_i]
        closed = ordered[n_a + n_i : n_a + n_i + plan.n_closed_book]

        chunks: list[tuple[MedicalRecord, Chunk]] = [(r, c) for r in pool for c in chunk_record(r)]
        first_chunk: dict[str, Chunk] = {}
        for r, c in chunks:
            first_chunk.setdefault(r.record_id, c)
        doc = self.embedder.encode_documents([index_text(c, r, "question_answer") for r, c in chunks])
        rag_q = rag_a + rag_i
        qv = self.embedder.encode_queries([r.question for r in rag_q]) if rag_q else np.zeros((0, doc.shape[1]))
        top = np.argsort(-(qv @ doc.T), axis=1)[:, : plan.candidates] if rag_q else np.zeros((0, 0), dtype=int)

        examples: list[SFTExample] = []
        id_rows: list[dict[str, Any]] = []
        for qi, q in enumerate(rag_q):
            answerable = qi < len(rag_a)
            ex, ev_ids = self._rag_example(
                q, [chunks[int(j)] for j in top[qi]], first_chunk, answerable, plan, rng, report
            )
            if ex is None:
                continue
            fmt: Format = "rag_answerable" if answerable else "rag_insufficient"
            report.counts[fmt] += 1
            examples.append(ex)
            id_rows.append(
                {
                    "record_id": q.record_id,
                    "split_group_id": q.split_group_id,
                    "format": fmt,
                    "evidence_record_ids": ev_ids,
                    "truncated": ex.truncated,
                }
            )
        closed_ex, closed_report = self.closed.build(closed)
        report.closed_book_report = closed_report.to_dict()
        report.counts["closed_book"] = len(closed_ex)
        for ex in closed_ex:
            examples.append(ex)
            id_rows.append(
                {
                    "record_id": ex.record_id,
                    "split_group_id": ex.split_group_id,
                    "format": "closed_book",
                    "evidence_record_ids": [],
                    "truncated": ex.truncated,
                }
            )
        return examples, report, id_rows

    def _rag_example(
        self,
        q: MedicalRecord,
        hits: list[tuple[MedicalRecord, Chunk]],
        first_chunk: dict[str, Chunk],
        answerable: bool,
        plan: MixPlan,
        rng: random.Random,
        report: MixReport,
    ) -> tuple[SFTExample | None, list[str]]:
        gold = first_chunk.get(q.record_id)
        if gold is None:
            report.skipped["no_gold_chunk"] += 1
            return None, []
        topic = (q.topic or "").casefold()
        seen: set[str] = set()
        picked: list[tuple[MedicalRecord, Chunk]] = []
        need = plan.k - 1 if answerable else plan.k
        for r, c in hits:
            if r.record_id in seen or r.split_group_id == q.split_group_id:
                continue
            if not answerable and ((r.topic or "").casefold() == topic or _norm(c.text) == _norm(gold.text)):
                continue
            seen.add(r.record_id)
            picked.append((r, c))
            if len(picked) == need:
                break
        if len(picked) < need:
            report.skipped["too_few_distractors"] += 1
            return None, []

        gold_pos = rng.randrange(plan.k) if answerable else -1
        while True:  # drop the lowest-ranked distractor until the prompt + target fit (never the gold)
            ev = list(picked)
            if answerable:
                ev.insert(min(gold_pos, len(ev)), (q, gold))
            blocks = [
                EvidenceBlock(label=f"E{i + 1}", text=c.text, topic=r.topic, source=r.source_name)
                for i, (r, c) in enumerate(ev)
            ]
            prompt = self._prompt_ids(q.question, blocks)
            budget = self.max_seq_len - len(prompt)
            if answerable:
                label = f"E{[r.record_id for r, _ in ev].index(q.record_id) + 1}"
                limit = min(plan.max_target_tokens, budget - len(self.eot_ids))
                text, used, total = cited_target(gold.text, label, lambda t, lim=limit: len(self._ids(t)) <= lim)
                if not text and limit >= plan.max_target_tokens:
                    break  # first sentence alone exceeds the target cap; dropping evidence cannot help
            else:
                used = total = 1
                text = SENTINEL
                if len(self._ids(text)) + len(self.eot_ids) > budget:
                    text = ""
            if text or not picked:
                break
            picked.pop()
            report.distractors_dropped_for_length += 1
        if not text:
            report.skipped["target_does_not_fit"] += 1
            return None, []

        supplied = [
            RetrievalHit(
                record_id=r.record_id,
                rank=i + 1,
                score=0.0,
                retriever="sft-train-dense:qa",
                evidence_text=c.text,
                corpus_version="sft-build",
            )
            for i, (r, c) in enumerate(ev)
        ]
        check = validate_citations(text, supplied)
        if answerable and (check.invalid_ids or check.cited_record_ids != [q.record_id] or check.has_sentinel):
            raise ValueError(f"invalid answerable target for {q.record_id}: {check}")
        if not answerable and (check.cited_record_ids or not check.has_sentinel):
            raise ValueError(f"invalid sentinel target for {q.record_id}: {check}")

        target = self._ids(text) + self.eot_ids
        full_answer = len(self._ids(neutralize(gold.text))) if answerable else len(target) - len(self.eot_ids)
        if answerable:
            report.gold_position[[r.record_id for r, _ in ev].index(q.record_id) + 1] += 1
        report.rag_prompt_tokens.append(len(prompt))
        ex = SFTExample(
            record_id=q.record_id,
            split_group_id=q.split_group_id,
            source_name=q.source_name,
            input_ids=prompt + target,
            labels=[IGNORE_INDEX] * len(prompt) + target,
            truncated=used < total,
            answer_tokens_full=full_answer,
            answer_tokens_kept=len(target) - len(self.eot_ids),
        )
        return ex, [r.record_id for r, _ in ev]

    # ------------------------------------------------------------------ sft-mix-v2b
    def _build_v2b(
        self, records: Sequence[MedicalRecord], plan: MixPlan
    ) -> tuple[list[SFTExample], MixReport, list[dict[str, Any]]]:
        report = MixReport(mix_version=plan.mix_version, plan=plan.as_dict())
        pool = [r for r in records if not EXCLUDED_FLAGS.intersection(r.quality_flags) and r.answer.strip()]
        ordered = sorted(pool, key=lambda r: _order_key(plan.seed, r.record_id))
        ctx = _Ctx.make(pool, plan)

        # format assignment in seeded order; each question gets at most one format
        n_a, n_info = plan.n_rag_answerable, plan.n_sentinel_info
        n_same = plan.n_rag_insufficient - n_info
        answerable = ordered[:n_a]
        sent_same: list[MedicalRecord] = []
        sent_info: list[MedicalRecord] = []
        rest: list[MedicalRecord] = []
        assign_skips: Counter[str] = Counter()
        for r in ordered[n_a:]:
            qt = r.question_type or ""
            if qt == plan.info_qtype and len(sent_info) < n_info:
                sent_info.append(r)
                continue
            if len(sent_same) < n_same:
                if qt in plan.sentinel_disallowed_qtypes:
                    assign_skips["sentinel_disallowed_qtype"] += 1
                elif not plan.sentinel_same_topic or ctx.candidates(r, sentinel=True)[0]:
                    sent_same.append(r)
                    continue
                else:
                    assign_skips["sentinel_no_candidates_after_guards"] += 1
            rest.append(r)
        closed = rest[: plan.n_closed_book]

        jobs = [(q, "answerable") for q in answerable] + [(q, plan.sentinel_kind) for q in sent_same]
        jobs += [(q, "sentinel_info_offtopic") for q in sent_info]
        doc = self.embedder.encode_documents([index_text(c, r, "question_answer") for r, c in ctx.chunks])
        qv = self.embedder.encode_queries([q.question for q, _ in jobs]) if jobs else np.zeros((0, doc.shape[1]))

        examples: list[SFTExample] = []
        id_rows: list[dict[str, Any]] = []
        guard_drops: dict[str, Counter[str]] = {}
        g7_pairs: Counter[str] = Counter()
        for i, (q, kind) in enumerate(jobs):
            ex, row = self._rag_example_v2b(ctx, q, kind, np.asarray(doc @ qv[i]), report, guard_drops, g7_pairs)
            if ex is None or row is None:
                continue
            report.counts[row["format"]] += 1
            examples.append(ex)
            id_rows.append(row)
        closed_ex, closed_report = self.closed.build(closed)
        report.closed_book_report = closed_report.to_dict()
        report.counts["closed_book"] = len(closed_ex)
        for ex in closed_ex:
            examples.append(ex)
            id_rows.append(
                {
                    "record_id": ex.record_id,
                    "split_group_id": ex.split_group_id,
                    "question_type": ctx.qtype[ex.record_id],
                    "format": "closed_book",
                    "sentinel_kind": None,
                    "evidence_record_ids": [],
                    "evidence_chunk_ids": [],
                    "truncated": ex.truncated,
                }
            )
        assigned = {
            "answerable": len(answerable),
            plan.sentinel_kind: len(sent_same),
            "sentinel_info_offtopic": len(sent_info),
            "closed_book": len(closed),
        }
        report.v2b = v2b_summary(id_rows, guard_drops, g7_pairs, assign_skips, plan, assigned)
        return examples, report, id_rows

    def _rag_example_v2b(
        self,
        ctx: _Ctx,
        q: MedicalRecord,
        kind: str,
        scores: np.ndarray,
        report: MixReport,
        guard_drops: dict[str, Counter[str]],
        g7_pairs: Counter[str],
    ) -> tuple[SFTExample | None, dict[str, Any] | None]:
        plan = ctx.plan
        answerable = kind == "answerable"
        rng_q = random.Random(f"{plan.seed}:v2b:{q.record_id}")  # noqa: S311 - per-question, order-independent
        weights = plan.same_topic_weights or [1.0]
        drawn_m = rng_q.choices(range(len(weights)), weights=weights)[0]
        if answerable:
            drawn, cap = drawn_m, plan.k - 1
        elif kind == "sentinel_same_topic":
            drawn, cap = drawn_m + 1, plan.k  # total on-topic T = m + 1, same weights as gold + m
        else:  # information or (v2c) all sentinels: off-topic evidence only
            drawn, cap = 0, 0
        gold = ctx.first_chunk[q.record_id]
        cands: dict[str, list[int]] = {}
        if kind == "sentinel_offtopic":  # no candidates are generated, so every guard counts 0 for this class
            guard_drops.setdefault(kind, Counter())
        elif kind != "sentinel_info_offtopic":
            cands, drops, pairs = ctx.candidates(q, sentinel=not answerable)
            guard_drops.setdefault(kind, Counter()).update(drops)
            g7_pairs.update(pairs)
        ranked = sorted(
            (
                (max(float(scores[ci]) for ci in cis), rid, max(cis, key=lambda ci: float(scores[ci])))
                for rid, cis in cands.items()
            ),
            reverse=True,
        )
        n_take = min(drawn, len(ranked), cap)
        if kind == "sentinel_same_topic" and plan.sentinel_selection == "random":
            chosen = sorted(rng_q.sample(ranked, n_take), reverse=True)
        elif plan.sentinel_selection in ("top_score", "random"):
            chosen = ranked[:n_take]
        else:
            raise ValueError(f"unknown sentinel_selection {plan.sentinel_selection!r}")
        same = [(score, ctx.chunks[ci]) for score, _, ci in chosen]

        tkq = topic_key(q.topic)
        gold_norm = _norm(gold.text)
        need_off = plan.k - (1 if answerable else 0) - len(same)
        seen = {q.record_id} | {pair[0].record_id for _, pair in same}
        off: list[tuple[float, tuple[MedicalRecord, Chunk]]] = []
        for j in np.argsort(-scores)[: plan.candidates]:
            if len(off) == need_off:
                break
            r, c = ctx.chunks[int(j)]
            if r.record_id in seen or r.split_group_id == q.split_group_id or _norm(c.text) == gold_norm:
                continue
            if tkq is not None and topic_key(r.topic) == tkq:
                raise ValueError(
                    f"off-group block {r.record_id} is on-topic for {q.record_id}: grouping invariant broken"
                )
            seen.add(r.record_id)
            off.append((float(scores[int(j)]), (r, c)))
        if len(off) < need_off:
            report.skipped["too_few_distractors"] += 1
            return None, None
        n_off_initial = len(off)

        gold_slot = rng_q.randrange(plan.k) if answerable else -1
        dropped: Counter[str] = Counter()
        while True:  # length drops: lowest-scored off-group block first, then lowest-scored same-topic; never gold
            ev = [pair for _, pair in sorted(same + off, key=lambda t: -t[0])]
            if answerable:
                ev.insert(min(gold_slot, len(ev)), (q, gold))
            text, used, total, prompt = self._fit_target(q, gold, ev, answerable, plan)
            if text is not None or not (same or off):
                break
            victims = off if off else same
            victims.remove(min(victims, key=lambda t: t[0]))
            dropped["offgroup" if victims is off else "same_topic"] += 1
            report.distractors_dropped_for_length += 1
        if not text:
            report.skipped["target_does_not_fit"] += 1
            return None, None

        ex = self._finish(q, gold, ev, text, used, total, prompt, answerable, report)
        on_topic = [tkq is not None and topic_key(r.topic) == tkq for r, _ in ev]
        fmt: Format = "rag_answerable" if answerable else "rag_insufficient"
        row = {
            "record_id": q.record_id,
            "split_group_id": q.split_group_id,
            "question_type": q.question_type,
            "format": fmt,
            "sentinel_kind": None if answerable else kind,
            "evidence_record_ids": [r.record_id for r, _ in ev],
            "evidence_chunk_ids": [c.chunk_id for _, c in ev],
            "evidence_on_topic": on_topic,
            "on_topic_total": sum(on_topic),
            "gold_slot": [r.record_id for r, _ in ev].index(q.record_id) + 1 if answerable else None,
            "same_topic_drawn": drawn,
            "same_topic_available_after_guards": len(ranked),
            "same_topic_realised_after_guards": n_take,
            "same_topic_realised_final": len(same),
            "n_offgroup_initial": n_off_initial,
            "n_offgroup_final": len(off),
            "dropped_for_length": dict(dropped),
            "truncated": ex.truncated,
        }
        return ex, row

    def _fit_target(
        self, q: MedicalRecord, gold: Chunk, ev: list[tuple[MedicalRecord, Chunk]], answerable: bool, plan: MixPlan
    ) -> tuple[str | None, int, int, list[int]]:
        """(text, used, total, prompt). text None = does not fit (drop evidence); '' = cannot ever fit."""
        blocks = [
            EvidenceBlock(label=f"E{i + 1}", text=c.text, topic=r.topic, source=r.source_name)
            for i, (r, c) in enumerate(ev)
        ]
        prompt = self._prompt_ids(q.question, blocks)
        budget = self.max_seq_len - len(prompt)
        if answerable:
            label = f"E{[r.record_id for r, _ in ev].index(q.record_id) + 1}"
            limit = min(plan.max_target_tokens, budget - len(self.eot_ids))
            text, used, total = cited_target(gold.text, label, lambda t, lim=limit: len(self._ids(t)) <= lim)
            if not text:
                return ("" if limit >= plan.max_target_tokens else None), used, total, prompt
            return text, used, total, prompt
        if len(self._ids(SENTINEL)) + len(self.eot_ids) > budget:
            return None, 1, 1, prompt
        return SENTINEL, 1, 1, prompt

    def _finish(
        self,
        q: MedicalRecord,
        gold: Chunk,
        ev: list[tuple[MedicalRecord, Chunk]],
        text: str,
        used: int,
        total: int,
        prompt: list[int],
        answerable: bool,
        report: MixReport,
    ) -> SFTExample:
        supplied = [
            RetrievalHit(
                record_id=r.record_id,
                rank=i + 1,
                score=0.0,
                retriever="sft-train-dense:qa",
                evidence_text=c.text,
                corpus_version="sft-build",
            )
            for i, (r, c) in enumerate(ev)
        ]
        check = validate_citations(text, supplied)
        if answerable and (check.invalid_ids or check.cited_record_ids != [q.record_id] or check.has_sentinel):
            raise ValueError(f"invalid answerable target for {q.record_id}: {check}")
        if not answerable and (check.cited_record_ids or not check.has_sentinel):
            raise ValueError(f"invalid sentinel target for {q.record_id}: {check}")
        target = self._ids(text) + self.eot_ids
        full_answer = len(self._ids(neutralize(gold.text))) if answerable else len(target) - len(self.eot_ids)
        if answerable:
            report.gold_position[[r.record_id for r, _ in ev].index(q.record_id) + 1] += 1
        report.rag_prompt_tokens.append(len(prompt))
        return SFTExample(
            record_id=q.record_id,
            split_group_id=q.split_group_id,
            source_name=q.source_name,
            input_ids=prompt + target,
            labels=[IGNORE_INDEX] * len(prompt) + target,
            truncated=used < total,
            answer_tokens_full=full_answer,
            answer_tokens_kept=len(target) - len(self.eot_ids),
        )


@dataclass
class _Ctx:
    """Per-split lookup tables for v2b candidate generation."""

    plan: MixPlan
    chunks: list[tuple[MedicalRecord, Chunk]]
    chunks_of: dict[str, list[int]]
    first_chunk: dict[str, Chunk]
    by_group_topic: dict[tuple[str, str], list[MedicalRecord]]
    featurizer: LexicalFeaturizer
    qtype: dict[str, str | None]
    _cache: dict[tuple[str, bool], tuple[dict[str, list[int]], Counter[str], Counter[str]]] = field(
        default_factory=dict
    )

    @classmethod
    def make(cls, pool: list[MedicalRecord], plan: MixPlan) -> _Ctx:
        chunks = [(r, c) for r in pool for c in chunk_record(r)]
        chunks_of: dict[str, list[int]] = defaultdict(list)
        for i, (r, _) in enumerate(chunks):
            chunks_of[r.record_id].append(i)
        by_group_topic: dict[tuple[str, str], list[MedicalRecord]] = defaultdict(list)
        for r in pool:
            tk = topic_key(r.topic)
            if tk is not None:
                by_group_topic[(r.split_group_id, tk)].append(r)
        return cls(
            plan=plan,
            chunks=chunks,
            chunks_of=dict(chunks_of),
            first_chunk={rid: chunks[ix[0]][1] for rid, ix in chunks_of.items()},
            by_group_topic=dict(by_group_topic),
            featurizer=LexicalFeaturizer.fit(c.text for _, c in chunks),  # this split's pool only
            qtype={r.record_id: r.question_type for r in pool},
        )

    def _coverage(self, question: str, text: str) -> float:
        return self.featurizer.features(question, text)[1]  # q_coverage_idf

    def candidates(
        self, q: MedicalRecord, *, sentinel: bool
    ) -> tuple[dict[str, list[int]], Counter[str], Counter[str]]:
        """Same-topic candidate chunks per record after guards G1-G7, plus drop counts (candidate chunks).

        G1 same question_type; G2 gold's duplicate group; G3 EXCLUDED_FLAGS; G4 text equal to gold chunk;
        G5 content-token Jaccard with gold chunk >= jaccard_max; G6 (sentinels only) idf coverage of the
        question >= gold's; G7 (sentinels only) question-type pair exclusion.
        Each chunk is counted at its first failing guard."""
        key = (q.record_id, sentinel)
        if key in self._cache:
            return self._cache[key]
        out: dict[str, list[int]] = {}
        drops: Counter[str] = Counter()
        pairs: Counter[str] = Counter()
        tk = topic_key(q.topic)
        gold = self.first_chunk.get(q.record_id)
        if tk is None or gold is None:
            self._cache[key] = (out, drops, pairs)
            return out, drops, pairs
        gold_norm = _norm(gold.text)
        gold_set = set(content(tokens(gold.text)))
        gold_cov = self._coverage(q.question, gold.text)
        excluded = self.plan.sentinel_excluded_qtypes.get(q.question_type or "", [])
        for r in self.by_group_topic.get((q.split_group_id, tk), []):
            if r.record_id == q.record_id:
                continue
            idx = self.chunks_of.get(r.record_id, [])
            if r.question_type == q.question_type:
                drops["G1_same_qtype"] += len(idx)
                continue
            if r.duplicate_group_id == q.duplicate_group_id:
                drops["G2_gold_dup_group"] += len(idx)
                continue
            if EXCLUDED_FLAGS.intersection(r.quality_flags):
                drops["G3_excluded_flags"] += len(idx)
                continue
            passing = []
            for ci in idx:
                text = self.chunks[ci][1].text
                c_set = set(content(tokens(text)))
                union = gold_set | c_set
                if _norm(text) == gold_norm:
                    drops["G4_text_equal_gold"] += 1
                elif union and len(gold_set & c_set) / len(union) >= self.plan.jaccard_max:
                    drops["G5_gold_jaccard"] += 1
                elif sentinel and self._coverage(q.question, text) >= gold_cov:  # answerable prompts are exempt
                    drops["G6_coverage_ge_gold"] += 1
                else:
                    passing.append(ci)
            if passing and sentinel and ("*" in excluded or r.question_type in excluded):
                drops["G7_sentinel_qtype_pair"] += len(passing)
                pairs[f"{q.question_type}->{r.question_type}"] += len(passing)
                continue
            if passing:
                out[r.record_id] = passing
        self._cache[key] = (out, drops, pairs)
        return out, drops, pairs


def _hist(values: Sequence[int], cap: int = 3) -> dict[str, int]:
    h: Counter[str] = Counter(str(v) if v < cap else f"{cap}+" for v in values)
    return dict(sorted(h.items()))


def cue_check(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """How well on_topic_total alone predicts answer vs refuse (best per-value rule) vs the majority baseline."""
    by_value: dict[int, Counter[str]] = defaultdict(Counter)
    for r in rows:
        by_value[int(r["on_topic_total"])][r["format"]] += 1
    n = sum(sum(c.values()) for c in by_value.values())
    if n == 0:
        return {"n": 0}
    totals: Counter[str] = Counter()
    for c in by_value.values():
        totals.update(c)
    return {
        "n": n,
        "rule_accuracy": sum(max(c.values()) for c in by_value.values()) / n,
        "majority_baseline": max(totals.values()) / n,
        "counts_by_on_topic_total": {str(v): dict(c) for v, c in sorted(by_value.items())},
    }


def _plan_vs_realised(
    rows: Sequence[dict[str, Any]], plan: MixPlan | None, assigned: dict[str, int] | None
) -> dict[str, dict[str, int | None]]:
    """planned (config) -> assigned (after seeded assignment) -> realised (after post-assignment skips, which
    are not replaced, so realised may be below plan)."""
    sentinel_kind = plan.sentinel_kind if plan is not None else "sentinel_same_topic"
    kinds = {
        "answerable": lambda r: r["format"] == "rag_answerable",
        sentinel_kind: lambda r: r["sentinel_kind"] == sentinel_kind,
        "sentinel_info_offtopic": lambda r: r["sentinel_kind"] == "sentinel_info_offtopic",
        "closed_book": lambda r: r["format"] == "closed_book",
    }
    planned = (
        {
            "answerable": plan.n_rag_answerable,
            sentinel_kind: plan.n_rag_insufficient - plan.n_sentinel_info,
            "sentinel_info_offtopic": plan.n_sentinel_info,
            "closed_book": plan.n_closed_book,
        }
        if plan is not None
        else {}
    )
    return {
        k: {
            "planned": planned.get(k),
            "assigned": (assigned or {}).get(k),
            "realised": sum(1 for r in rows if pred(r)),
        }
        for k, pred in kinds.items()
    }


def v2b_summary(
    rows: Sequence[dict[str, Any]],
    guard_drops: dict[str, Counter[str]],
    g7_pairs: Counter[str],
    assign_skips: Counter[str],
    plan: MixPlan | None = None,
    assigned: dict[str, int] | None = None,
) -> dict[str, Any]:
    """IDs/counts-only diagnostics for the step-3 report (all recomputable from the manifest rows).

    Produced once per build (train and validation_rag have separate reports, hence separate plan-vs-realised)."""
    rag = [r for r in rows if r["format"] != "closed_book"]
    ans = [r for r in rag if r["format"] == "rag_answerable"]
    sen = [r for r in rag if r["format"] == "rag_insufficient"]
    nonzero = [r for r in rag if r["same_topic_available_after_guards"] > 0]
    sentinel_qtypes = sorted({str(r["question_type"]) for r in sen})
    classes = {"rag_answerable": ans, "rag_insufficient": sen}
    sentinel_kinds = ["sentinel_same_topic", "sentinel_info_offtopic"]
    if any(r["sentinel_kind"] == "sentinel_offtopic" for r in sen) or (
        plan is not None and not plan.sentinel_same_topic
    ):
        sentinel_kinds.insert(1, "sentinel_offtopic")
    for kind in sentinel_kinds:
        classes[kind] = [r for r in sen if r["sentinel_kind"] == kind]
    return {
        "assignment_skips": dict(assign_skips),
        "guard_drops_candidate_chunks": {k: {g: v.get(g, 0) for g in GUARDS} for k, v in sorted(guard_drops.items())},
        "g7_drops_by_pair": dict(sorted(g7_pairs.items())),
        "qtype_mix_by_format": {
            f: dict(sorted(Counter(str(r["question_type"]) for r in rows if r["format"] == f).items()))
            for f in ("rag_answerable", "rag_insufficient", "closed_book")
        },
        "on_topic_total_hist": {
            k: {str(n): sum(r["on_topic_total"] == n for r in v) for n in range(6)} for k, v in classes.items()
        },
        "sentinel_same_topic_zero_on_topic_after_length_drops": sum(
            r["on_topic_total"] == 0 for r in classes["sentinel_same_topic"]
        ),
        "plan_vs_realised": _plan_vs_realised(rows, plan, assigned),
        "same_topic_drawn_hist": {k: _hist([r["same_topic_drawn"] for r in v], 5) for k, v in classes.items()},
        "same_topic_realised_final_hist": {
            k: _hist([r["same_topic_realised_final"] for r in v], 5) for k, v in classes.items()
        },
        "zero_candidate": {k: sum(r["same_topic_available_after_guards"] == 0 for r in v) for k, v in classes.items()},
        "zero_candidate_by_sentinel_qtype": dict(
            sorted(Counter(str(r["question_type"]) for r in sen if r["same_topic_available_after_guards"] == 0).items())
        ),
        "length_drops": dict(sum((Counter(r["dropped_for_length"]) for r in rag), Counter())),
        "cue_check": {
            "all": cue_check(rag),
            "excluding_zero_candidate": cue_check(nonzero),
            "per_sentinel_qtype": {
                qt: cue_check([r for r in rag if str(r["question_type"]) == qt]) for qt in sentinel_qtypes
            },
        },
    }


def mixed_data(
    train_path: Path,
    val_path: Path,
    embedder: DocEmbedder,
    train_plan: MixPlan,
    val_rag_plan: MixPlan,
    *,
    max_seq_len: int,
    closed_book_builder: MessageBuilder,
    closed_val_max: int,
) -> Any:
    """Data factory for ``lora.run_sft``: mixed train set; validation loss on closed-book AND RAG examples.

    Train and validation are built independently, each with its own in-split evidence pool.
    """
    from medquad_qa.training.lora import PreparedData

    def factory(tokenizer: Any) -> PreparedData:
        builder = MixedSFTBuilder(tokenizer, embedder, max_seq_len=max_seq_len, closed_book_builder=closed_book_builder)
        train_ex, train_report, rows = builder.build(list(read_records(train_path)), train_plan)
        val_records = list(read_records(val_path))
        val_rag, val_rag_report, val_rows = builder.build(val_records, val_rag_plan)
        val_closed, val_closed_report = builder.closed.build(val_records)
        return PreparedData(
            train=train_ex,
            val_sets={"eval": val_closed[:closed_val_max], "eval_rag": val_rag},
            reports={
                "train": train_report.to_dict(),
                "val_rag": val_rag_report.to_dict(),
                "val": val_closed_report.to_dict(),
            },
            id_rows=rows,
            train_sha256=file_sha256(train_path),
            val_id_rows=val_rows
            + [
                {
                    "record_id": e.record_id,
                    "split_group_id": e.split_group_id,
                    "format": "closed_book",
                    "evidence_record_ids": [],
                    "truncated": e.truncated,
                }
                for e in val_closed[:closed_val_max]
            ],
        )

    return factory
