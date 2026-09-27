"""Offline test doubles implementing the domain ports.

These fakes let the domain/application layers be tested without any network
access or Azure credentials.
"""

from __future__ import annotations

from collections.abc import Sequence

from enterprise_rag.domain.models import ChatMessage, RetrievedChunk


class FakeEmbeddingProvider:
    """Deterministic, dependency-free embedding provider for tests."""

    def __init__(self, dimensions: int = 16) -> None:
        self.dimensions = dimensions
        self.embedded_texts: list[str] = []

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        self.embedded_texts.extend(texts)
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for index, token in enumerate(text.split()):
            vector[index % self.dimensions] += float(len(token))
        return vector


class FakeChatModel:
    """Chat model that echoes a canned reply and records the last prompt."""

    def __init__(self, reply: str = "a synthesized answer") -> None:
        self.reply = reply
        self.last_messages: list[ChatMessage] | None = None

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        self.last_messages = list(messages)
        return self.reply


class FakeRetriever:
    """Retriever that returns preset results and records its calls."""

    def __init__(self, results: Sequence[RetrievedChunk]) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        self.calls.append((query, top_k))
        return self.results[:top_k]
