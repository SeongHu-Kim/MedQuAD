"""TEMPORARY x86-vs-aarch64 diagnostic for the two SFT golden-hash tests (D-079 follow-up; remove in D-080).

Rebuilds the v2 and v2b test fixtures exactly as tests/training/test_training_sft_rag_data.py does and prints
DIAG lines to compare between the CI runners: versions, tokenizer hash, embedding and score-matrix hashes,
whether np.argsort's default kind agrees with a stable sort on the builder's calls, and per-example evidence.
Synthetic fixture data only. Run from the repo root: python scripts/ci/golden_diag.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "training"))

import test_training_sft_rag_data as T  # noqa: E402

from medquad_qa.models.testing import TINY_TOKENIZER_JSON, build_tiny_tokenizer  # noqa: E402
from medquad_qa.retrieval.embedding import HashingEmbedder  # noqa: E402
from medquad_qa.training.sft_rag_data import MixedSFTBuilder  # noqa: E402


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def out(tag: str, **kw: Any) -> None:
    print(f"DIAG {tag} " + json.dumps(kw, sort_keys=True, default=str), flush=True)


class RecordingEmbedder(HashingEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, np.ndarray]] = []

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        m = super().encode_documents(texts)
        self.calls.append(("doc", m))
        return m

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        m = super().encode_queries(texts)
        self.calls.append(("query", m))
        return m


ARGSORT = np.argsort
STATS: dict[str, int] = {}


def recording_argsort(a: Any, axis: int = -1, kind: Any = None, **kw: Any) -> Any:
    """Return numpy's default result unchanged; count where it differs from a stable sort in the top 20."""
    res = ARGSORT(a, axis=axis, kind=kind, **kw)
    stable = ARGSORT(a, axis=axis, kind="stable")
    arr = np.asarray(a)
    STATS["calls"] = STATS.get("calls", 0) + 1
    rows_r = res.reshape(-1, res.shape[-1]) if res.ndim > 1 else res[None, :]
    rows_s = stable.reshape(-1, stable.shape[-1]) if stable.ndim > 1 else stable[None, :]
    rows_a = arr.reshape(-1, arr.shape[-1]) if arr.ndim > 1 else arr[None, :]
    for r, s, v in zip(rows_r, rows_s, rows_a, strict=True):
        STATS["rows"] = STATS.get("rows", 0) + 1
        if not np.array_equal(r[:20], s[:20]):
            STATS["rows_top20_default_ne_stable"] = STATS.get("rows_top20_default_ne_stable", 0) + 1
        top = v[s[:20]]
        STATS["exact_ties_in_top20"] = STATS.get("exact_ties_in_top20", 0) + int(np.sum(np.diff(top) == 0))
        STATS["near_ties_in_top20"] = STATS.get("near_ties_in_top20", 0) + int(
            np.sum((np.abs(np.diff(top)) < 1e-6) & (np.diff(top) != 0))
        )
    return res


def run(name: str, tok: Any, build: Any) -> None:
    STATS.clear()
    emb = RecordingEmbedder()
    examples, report, rows = build(tok, emb)
    full, rows_h, ids_h = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    for ex, row in zip(examples, rows, strict=True):
        full.update(json.dumps([row, ex.input_ids, ex.labels], sort_keys=True).encode())
        rows_h.update(json.dumps(row, sort_keys=True).encode())
        ids_h.update(json.dumps([ex.input_ids, ex.labels]).encode())
    out(
        f"{name}.hash",
        full=full.hexdigest(),
        rows_only=rows_h.hexdigest()[:16],
        tokens_only=ids_h.hexdigest()[:16],
        report=sha(json.dumps(report.to_dict(), sort_keys=True, default=str).encode()),
        n_examples=len(examples),
    )
    mats = dict(emb.calls)
    doc, qv = mats["doc"], mats["query"]
    sc = qv @ doc.T
    sc_loop = np.stack([np.asarray(doc @ qv[i]) for i in range(len(qv))]) if len(qv) else sc
    out(
        f"{name}.matrices",
        doc=sha(doc.tobytes()),
        query=sha(qv.tobytes()),
        scores_matmul=sha(sc.tobytes()),
        scores_rowwise=sha(sc_loop.tobytes()),
        scores_f64=sha(np.round(qv.astype(np.float64) @ doc.astype(np.float64).T, 6).tobytes()),
    )
    out(f"{name}.argsort", **STATS)
    for i, sq in enumerate(sc[:5]):
        out(
            f"{name}.top5.q{i}",
            default=ARGSORT(-sq)[:5].tolist(),
            stable=ARGSORT(-sq, kind="stable")[:5].tolist(),
            scores=[float(x) for x in np.sort(-sq)[:5] * -1],
        )
    for i, (ex, row) in enumerate(zip(examples[:3], rows[:3], strict=True)):
        out(
            f"{name}.ex{i}",
            format=row.get("format"),
            evidence=row.get("evidence_record_ids"),
            input_ids=ex.input_ids[:40],
            first_target_labels=[t for t in ex.labels if t != -100][:40],
        )


def main() -> None:
    import tokenizers
    import transformers

    np_cfg = np.show_config(mode="dicts")
    out(
        "env",
        machine=platform.machine(),
        python=platform.python_version(),
        numpy=np.__version__,
        tokenizers=tokenizers.__version__,
        transformers=transformers.__version__,
        blas=np_cfg.get("Build Dependencies", {}).get("blas"),
        simd=np_cfg.get("SIMD Extensions"),
    )
    tok = build_tiny_tokenizer()
    out(
        "tokenizer",
        json_file=sha(TINY_TOKENIZER_JSON.read_bytes()),
        loaded=sha(tok.backend_tokenizer.to_str().encode()),
        vocab=len(tok),
        probe=tok.encode("What are the symptoms of cond3x ?", add_special_tokens=False),
    )
    np.argsort = recording_argsort  # type: ignore[assignment]  # the builder calls np.argsort via the module
    try:
        run(
            "v2",
            tok,
            lambda t, e: MixedSFTBuilder(t, e, max_seq_len=4096).build(T._records("tr", 15, 0), T.PLAN),
        )
        run(
            "v2b",
            tok,
            lambda t, e: MixedSFTBuilder(t, e, max_seq_len=4096).build(T._v2b_records(), T._v2b_plan()),
        )
    finally:
        np.argsort = ARGSORT  # type: ignore[assignment]
    out("golden", v2=T.V2_GOLDEN_SHA256, v2b_rows=T.V2B_GOLDEN_ROWS_SHA256)


if __name__ == "__main__":
    main()
