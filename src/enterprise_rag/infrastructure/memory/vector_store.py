"""In-memory vector store adapter.

Useful for local development and for tests that must run without network access
or Azure credentials. Not intended for production-scale corpora.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from ...domain.models import EmbeddedChunk, RetrievedChunk

__all__ = ["InMemoryVectorStore"]


class InMemoryVectorStore:
    """A trivial cosine-similarity vector store backed by a Python list."""

    def __init__(self) -> None:
        self._items: list[EmbeddedChunk] = []

    def add(self, chunks: Sequence[EmbeddedChunk]) -> None:
        self._items.extend(chunks)

    def search(self, embedding: Sequence[float], top_k: int) -> list[RetrievedChunk]:
        if top_k <= 0 or not self._items:
            return []
        scored = [
            RetrievedChunk(chunk=item.chunk, score=_cosine_similarity(embedding, item.embedding))
            for item in self._items
        ]
        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self._items)


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
