"""medquad_qa.rag: grounded QA orchestration (langchain-core), prompts, citation validation, abstention.

``medquad_qa.rag.prompts`` is pure (no langchain/torch) so training code can import the frozen templates;
the pipeline and factory are imported lazily.
"""

from typing import Any

from medquad_qa.rag.prompts import (
    CLOSED_BOOK_PROMPT_VERSION,
    PROMPT_VERSION,
    SENTINEL,
    build_closed_book_messages,
    build_rag_messages,
)

__all__ = [
    "CLOSED_BOOK_PROMPT_VERSION",
    "PROMPT_VERSION",
    "SENTINEL",
    "RagPipeline",
    "RagSettings",
    "build_closed_book_messages",
    "build_pipeline",
    "build_rag_messages",
]


def __getattr__(name: str) -> Any:
    if name == "build_pipeline":
        from medquad_qa.rag.factory import build_pipeline

        return build_pipeline
    if name == "RagPipeline":
        from medquad_qa.rag.pipeline import RagPipeline

        return RagPipeline
    if name == "RagSettings":
        from medquad_qa.rag.settings import RagSettings

        return RagSettings
    raise AttributeError(name)
