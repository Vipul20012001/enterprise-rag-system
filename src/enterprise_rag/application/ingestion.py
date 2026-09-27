"""Ingestion use case: documents -> chunks -> embeddings -> vector store."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ..domain.models import Chunk, Document, EmbeddedChunk
from ..domain.ports import ChunkingStrategy, EmbeddingProvider, VectorStore

__all__ = ["IngestionService", "IngestionResult"]


@dataclass(frozen=True)
class IngestionResult:
    """Summary of an ingestion run."""

    documents: int
    chunks: int


class IngestionService:
    """Chunk documents, embed the chunks in batches, and persist them."""

    def __init__(
        self,
        chunker: ChunkingStrategy,
        embedder: EmbeddingProvider,
        store: VectorStore,
        *,
        batch_size: int = 64,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self._chunker = chunker
        self._embedder = embedder
        self._store = store
        self._batch_size = batch_size

    def ingest(self, documents: Iterable[Document]) -> IngestionResult:
        chunks: list[Chunk] = []
        document_count = 0
        for document in documents:
            document_count += 1
            chunks.extend(self._chunker.split(document))

        if not chunks:
            return IngestionResult(documents=document_count, chunks=0)

        embedded: list[EmbeddedChunk] = []
        for batch in _batched(chunks, self._batch_size):
            vectors = self._embedder.embed_texts([chunk.content for chunk in batch])
            embedded.extend(
                EmbeddedChunk(chunk=chunk, embedding=vector)
                for chunk, vector in zip(batch, vectors, strict=True)
            )

        self._store.add(embedded)
        return IngestionResult(documents=document_count, chunks=len(chunks))


def _batched(items: Sequence[Chunk], size: int) -> Iterable[Sequence[Chunk]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
