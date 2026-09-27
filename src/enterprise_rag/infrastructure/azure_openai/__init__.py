"""Azure OpenAI adapters."""

from __future__ import annotations

from .chat import AzureOpenAIChatModel
from .embeddings import AzureOpenAIEmbeddingProvider

__all__ = ["AzureOpenAIEmbeddingProvider", "AzureOpenAIChatModel"]
