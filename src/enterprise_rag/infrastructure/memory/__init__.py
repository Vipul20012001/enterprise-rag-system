"""In-memory / offline infrastructure adapters."""

from __future__ import annotations

from .embeddings import HashingEmbeddingProvider
from .vector_store import InMemoryVectorStore

__all__ = ["HashingEmbeddingProvider", "InMemoryVectorStore"]
