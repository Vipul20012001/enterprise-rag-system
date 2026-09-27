"""Application configuration via ``pydantic-settings``.

Environment variables use the ``RAG_`` prefix with ``__`` as the nested
delimiter, e.g. ``RAG_AZURE_OPENAI__ENDPOINT``. Values may also be supplied via a
local ``.env`` file (git-ignored).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "AzureOpenAISettings",
    "AzureSearchSettings",
    "ChunkingSettings",
    "RetrievalSettings",
    "AppSettings",
    "get_settings",
]


class AzureOpenAISettings(BaseModel):
    """Azure OpenAI connection and deployment settings."""

    endpoint: str
    api_key: str
    api_version: str = "2024-10-21"
    chat_deployment: str
    embedding_deployment: str


class AzureSearchSettings(BaseModel):
    """Azure AI Search connection and index settings."""

    endpoint: str
    api_key: str
    index_name: str = "research-papers"
    embedding_dimensions: int = 1536


class ChunkingSettings(BaseModel):
    """Chunking parameters for the Phase 1 fixed-size chunker."""

    chunk_size: int = 1000
    chunk_overlap: int = 200


class RetrievalSettings(BaseModel):
    """Retrieval parameters."""

    top_k: int = 5


class AppSettings(BaseSettings):
    """Root application settings."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    azure_openai: AzureOpenAISettings
    azure_search: AzureSearchSettings
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)


@lru_cache
def get_settings() -> AppSettings:
    """Return cached application settings loaded from the environment/.env."""
    return AppSettings()
