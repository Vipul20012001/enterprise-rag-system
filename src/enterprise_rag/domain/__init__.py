"""Domain layer: pure models, ports (interfaces), and errors."""

from __future__ import annotations

from .errors import (
    ConfigurationError,
    GenerationError,
    IngestionError,
    RagError,
    RetrievalError,
)
from .models import (
    ChatMessage,
    Chunk,
    Document,
    EmbeddedChunk,
    RagAnswer,
    RetrievedChunk,
    Role,
)
from .ports import (
    ChatModel,
    ChunkingStrategy,
    DocumentLoader,
    EmbeddingProvider,
    PromptBuilder,
    Retriever,
    VectorStore,
)

__all__ = [
    # models
    "Role",
    "ChatMessage",
    "Document",
    "Chunk",
    "EmbeddedChunk",
    "RetrievedChunk",
    "RagAnswer",
    # ports
    "EmbeddingProvider",
    "ChatModel",
    "VectorStore",
    "Retriever",
    "ChunkingStrategy",
    "PromptBuilder",
    "DocumentLoader",
    # errors
    "RagError",
    "ConfigurationError",
    "IngestionError",
    "RetrievalError",
    "GenerationError",
]
