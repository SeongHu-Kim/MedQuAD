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
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from medquad_qa.contracts.qa import RetrievalHit
from medquad_qa.contracts.records import MedicalRecord
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

    def to_dict(self) -> dict[str, Any]:
        p = sorted(self.rag_prompt_tokens)
        return {
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
        report = MixReport(mix_version=MIX_VERSION, plan=vars(plan).copy())
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
