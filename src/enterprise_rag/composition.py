"""Composition root.

The single place that knows about concrete infrastructure adapters and wires
them into the application services. Swapping a provider (Phase 2/3) means editing
only this module (or adding a sibling factory), not the domain/application code.
"""

from __future__ import annotations

from dataclasses import dataclass

from .application.chunking import FixedSizeChunker
from .application.ingestion import IngestionService
from .application.prompts import DefaultPromptBuilder
from .application.query import RagQueryService
from .application.retrieval import VectorRetriever
from .config.settings import AppSettings, get_settings
from .domain.ports import DocumentLoader, VectorStore
from .infrastructure.azure_openai.chat import AzureOpenAIChatModel
from .infrastructure.azure_openai.embeddings import AzureOpenAIEmbeddingProvider
from .infrastructure.azure_search.vector_store import AzureAISearchVectorStore
from .infrastructure.loaders.text_loader import TextFileLoader

__all__ = ["RagServices", "build_services"]


@dataclass(frozen=True)
class RagServices:
    """The fully wired services exposed to entry points (CLI, API, ...)."""

    ingestion: IngestionService
    query: RagQueryService
    loader: DocumentLoader
    store: VectorStore


def build_services(settings: AppSettings | None = None) -> RagServices:
    """Wire the Azure-backed Naive RAG pipeline from settings."""
    settings = settings or get_settings()
    azure_openai = settings.azure_openai
    azure_search = settings.azure_search

    embedder = AzureOpenAIEmbeddingProvider(
        endpoint=azure_openai.endpoint,
        api_key=azure_openai.api_key,
        api_version=azure_openai.api_version,
        deployment=azure_openai.embedding_deployment,
    )
    chat = AzureOpenAIChatModel(
        endpoint=azure_openai.endpoint,
        api_key=azure_openai.api_key,
        api_version=azure_openai.api_version,
        deployment=azure_openai.chat_deployment,
    )
    store = AzureAISearchVectorStore(
        endpoint=azure_search.endpoint,
        api_key=azure_search.api_key,
        index_name=azure_search.index_name,
        vector_dimensions=azure_search.embedding_dimensions,
    )

    chunker = FixedSizeChunker(
        chunk_size=settings.chunking.chunk_size,
        chunk_overlap=settings.chunking.chunk_overlap,
    )
    retriever = VectorRetriever(embedder=embedder, store=store)
    prompt_builder = DefaultPromptBuilder()

    ingestion = IngestionService(chunker=chunker, embedder=embedder, store=store)
    query = RagQueryService(
        retriever=retriever,
        chat=chat,
        prompt_builder=prompt_builder,
        top_k=settings.retrieval.top_k,
    )

    return RagServices(
        ingestion=ingestion,
        query=query,
        loader=TextFileLoader(),
        store=store,
    )
