"""Application layer: use cases and default strategies.

Depends only on domain ports, never on infrastructure/SDK types.
"""

from __future__ import annotations

from .chunking import FixedSizeChunker
from .ingestion import IngestionResult, IngestionService
from .prompts import DefaultPromptBuilder
from .query import RagQueryService
from .retrieval import VectorRetriever

__all__ = [
    "FixedSizeChunker",
    "DefaultPromptBuilder",
    "VectorRetriever",
    "IngestionService",
    "IngestionResult",
    "RagQueryService",
]
