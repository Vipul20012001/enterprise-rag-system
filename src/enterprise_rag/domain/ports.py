"""Domain ports: the interfaces the application depends on.

Every port is a :class:`typing.Protocol`, so any object with a matching shape
satisfies it (structural typing). Concrete implementations live in the
``infrastructure`` package; the domain and application layers depend only on
these abstractions. New providers/strategies are added by implementing a port,
never by editing the core.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .models import ChatMessage, Chunk, Document, EmbeddedChunk, RetrievedChunk

__all__ = [
    "EmbeddingProvider",
    "ChatModel",
    "VectorStore",
    "Retriever",
    "ChunkingStrategy",
    "PromptBuilder",
    "DocumentLoader",
]


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into dense vectors."""

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of documents/chunks."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        ...


@runtime_checkable
class ChatModel(Protocol):
    """Generates an assistant reply for a sequence of chat messages."""

    def complete(self, messages: Sequence[ChatMessage]) -> str: ...


@runtime_checkable
class VectorStore(Protocol):
    """Persists embedded chunks and performs nearest-neighbour search."""

    def add(self, chunks: Sequence[EmbeddedChunk]) -> None: ...

    def search(self, embedding: Sequence[float], top_k: int) -> list[RetrievedChunk]: ...


@runtime_checkable
class Retriever(Protocol):
    """Returns the most relevant chunks for a query.

    This is the primary Phase 2/3 extension point: hybrid search, reranking, and
    query rewriting are all implemented as alternative ``Retriever`` strategies.
    """

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]: ...


@runtime_checkable
class ChunkingStrategy(Protocol):
    """Splits a document into chunks."""

    def split(self, document: Document) -> list[Chunk]: ...


@runtime_checkable
class PromptBuilder(Protocol):
    """Builds the chat messages sent to the :class:`ChatModel`."""

    def build(self, question: str, contexts: Sequence[RetrievedChunk]) -> list[ChatMessage]: ...


@runtime_checkable
class DocumentLoader(Protocol):
    """Loads raw documents from a source (path, URI, connector, ...)."""

    def load(self, source: str) -> list[Document]: ...
