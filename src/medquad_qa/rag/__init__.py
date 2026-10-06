"""medquad_qa.rag: grounded QA orchestration (langchain-core), prompts, citation validation, abstention."""

from medquad_qa.rag.factory import build_pipeline
from medquad_qa.rag.pipeline import RagPipeline
from medquad_qa.rag.prompts import PROMPT_VERSION, SENTINEL, build_closed_book_messages, build_rag_messages
from medquad_qa.rag.settings import RagSettings

__all__ = [
    "PROMPT_VERSION",
    "SENTINEL",
    "RagPipeline",
    "RagSettings",
    "build_closed_book_messages",
    "build_pipeline",
    "build_rag_messages",
]
