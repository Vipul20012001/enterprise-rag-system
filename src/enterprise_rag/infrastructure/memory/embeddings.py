"""Deterministic, offline embedding provider.

A hashing-based embedder that requires no network or model weights. It exists so
the full pipeline can be exercised locally and in tests. It is **not** semantic
and must not be used for real retrieval quality.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

__all__ = ["HashingEmbeddingProvider"]


class HashingEmbeddingProvider:
    """Map text to a fixed-length vector via token hashing (bag-of-words)."""

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")
        self._dimensions = dimensions

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dimensions
            vector[bucket] += 1.0
        return vector
