"""Default retriever: embed the query, then nearest-neighbour search.

This is the Phase 1 baseline. Hybrid search, reranking, and query rewriting are
future strategies that implement the same :class:`Retriever` port.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..domain.models import RetrievedChunk
from ..domain.ports import EmbeddingProvider, VectorStore

__all__ = ["VectorRetriever"]


@dataclass(frozen=True)
class VectorRetriever:
    """Embed a query and return the closest chunks from the vector store."""

    embedder: EmbeddingProvider
    store: VectorStore

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if top_k <= 0:
            return []
        query_vector = self.embedder.embed_query(query)
        return self.store.search(query_vector, top_k)
