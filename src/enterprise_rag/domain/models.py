"""Provider-independent domain models.

These are pure Python value objects. They must never import third-party SDK types
(Azure, OpenAI, LangChain, etc.) so that the domain stays portable across
providers and phases.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "Role",
    "ChatMessage",
    "Document",
    "Chunk",
    "EmbeddedChunk",
    "RetrievedChunk",
    "RagAnswer",
]


class Role(str, Enum):
    """Role of a chat message participant."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """A single message in a chat completion request."""

    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class Document:
    """A source document (e.g. a research paper) prior to chunking."""

    id: str
    content: str
    source: str = ""
    title: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A contiguous slice of a :class:`Document`."""

    id: str
    document_id: str
    content: str
    index: int
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    """A :class:`Chunk` paired with its dense vector representation."""

    chunk: Chunk
    embedding: Sequence[float]


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A :class:`Chunk` returned from retrieval with a relevance score."""

    chunk: Chunk
    score: float


@dataclass(frozen=True, slots=True)
class RagAnswer:
    """The result of answering a question over retrieved context."""

    question: str
    answer: str
    sources: tuple[RetrievedChunk, ...] = ()
