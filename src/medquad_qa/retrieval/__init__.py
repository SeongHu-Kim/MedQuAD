"""medquad_qa.retrieval: BM25, dense (Qdrant) and hybrid retrievers implementing ``contracts.Retriever``.

Importing this package does not import torch or qdrant-client; heavy backends load lazily.
"""

from medquad_qa.retrieval.bm25 import BM25Retriever
from medquad_qa.retrieval.factory import RetrievalBundle, build_retriever, require_retriever
from medquad_qa.retrieval.fixture import FixtureRetriever
from medquad_qa.retrieval.hybrid import LEXICAL_FALLBACK, HybridRetriever, rrf_fuse
from medquad_qa.retrieval.settings import RetrievalSettings
from medquad_qa.retrieval.store import CorpusStore

__all__ = [
    "LEXICAL_FALLBACK",
    "BM25Retriever",
    "CorpusStore",
    "FixtureRetriever",
    "HybridRetriever",
    "RetrievalBundle",
    "RetrievalSettings",
    "build_retriever",
    "require_retriever",
    "rrf_fuse",
]
