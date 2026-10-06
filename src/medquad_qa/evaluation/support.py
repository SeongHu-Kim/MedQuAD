"""Claim-support scorers: P(evidence entails claim).

- ``LexicalSupportScorer``: deterministic content-word containment; offline fallback and test double.
  It is a weak proxy and is always reported under its own name, never as NLI.
- ``NLISupportScorer``: a pinned Hugging Face NLI cross-encoder (loaded lazily, local files only by default).
  Long evidence is split into sentence windows; the claim's score is the max entailment over windows.
"""

from __future__ import annotations

import re
import threading
from typing import Any, Protocol, runtime_checkable

from medquad_qa.evaluation.claims import split_sentences

_WORD = re.compile(r"[a-z0-9]+")
# Small closed stopword list; negations and quantity words are deliberately NOT stopwords.
_STOP_WORDS = (
    "a an the of to in on for and or is are was were be been being it its this that these those with by as at "
    "from can may might could would should will which who whom what when where how also than then there their "
    "they them he she his her you your we our us i about into such some any each other more most many much"
)
_STOP = frozenset(_STOP_WORDS.split())


@runtime_checkable
class SupportScorer(Protocol):
    name: str

    def score(self, evidence: str, claim: str) -> float:
        """Return a support score in [0, 1]; higher means the evidence supports the claim."""
        ...


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


class LexicalSupportScorer:
    """Fraction of the claim's content words present in the evidence."""

    name = "lexical_containment_v1"

    def score(self, evidence: str, claim: str) -> float:
        cw = _content_words(claim)
        if not cw:
            return 0.0
        return len(cw & _content_words(evidence)) / len(cw)


class NLISupportScorer:
    """Entailment probability from a sequence-classification NLI model with an ``entailment`` label.

    ``revision`` must be a commit hash; the scorer records ``name = '<model_id>@<revision>'``.
    """

    def __init__(
        self,
        model_id: str,
        revision: str,
        *,
        device: str | None = None,
        window_words: int = 300,
        local_files_only: bool = True,
        batch_size: int = 16,
    ) -> None:
        self.model_id, self.revision = model_id, revision
        self.name = f"{model_id}@{revision}"
        self.window_words = window_words
        self.local_files_only = local_files_only
        self.batch_size = batch_size
        self._device = device
        self._lock = threading.Lock()
        self._model: Any = None
        self._tok: Any = None
        self._entail_idx: int | None = None

    def _load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            from medquad_qa.models.torch_runtime import apply_native_jit_guard

            apply_native_jit_guard()  # torch 2.14 Triton paths need Python.h, absent on this host (model-engineer)
            kw ={"revision": self.revision, "local_files_only": self.local_files_only}
            tok = AutoTokenizer.from_pretrained(self.model_id, **kw)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_id, **kw)
            labels = {str(v).lower(): int(k) for k, v in model.config.id2label.items()}
            if "entailment" not in labels:
                raise ValueError(f"{self.name}: no 'entailment' label in {model.config.id2label}")
            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._device = device
            self._entail_idx = labels["entailment"]
            self._tok, self._model = tok, model.to(device).eval()

    def windows(self, evidence: str) -> list[str]:
        out: list[str] = []
        cur: list[str] = []
        n = 0
        for sent in split_sentences(evidence) or [evidence]:
            w = len(sent.split())
            if cur and n + w > self.window_words:
                out.append(" ".join(cur))
                cur, n = [], 0
            cur.append(sent)
            n += w
        if cur:
            out.append(" ".join(cur))
        return out

    def score_many(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Max-over-windows entailment probability for each (evidence, claim) pair."""
        self._load()
        import torch

        flat: list[tuple[int, str, str]] = [(i, w, c) for i, (e, c) in enumerate(pairs) for w in self.windows(e)]
        best = [0.0] * len(pairs)
        for s in range(0, len(flat), self.batch_size):
            chunk = flat[s : s + self.batch_size]
            enc = self._tok(
                [w for _, w, _ in chunk],
                [c for _, _, c in chunk],
                truncation="only_first",
                max_length=512,
                padding=True,
                return_tensors="pt",
            ).to(self._device)
            with torch.inference_mode():
                probs = self._model(**enc).logits.float().softmax(-1)[:, self._entail_idx].tolist()
            for (i, _, _), p in zip(chunk, probs, strict=True):
                best[i] = max(best[i], float(p))
        return best

    def score(self, evidence: str, claim: str) -> float:
        return self.score_many([(evidence, claim)])[0]
