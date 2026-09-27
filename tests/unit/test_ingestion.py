from __future__ import annotations

from fakes import FakeEmbeddingProvider

from enterprise_rag.application.chunking import FixedSizeChunker
from enterprise_rag.application.ingestion import IngestionService
from enterprise_rag.domain.models import Document
from enterprise_rag.infrastructure.memory.vector_store import InMemoryVectorStore


def test_ingest_counts_documents_and_chunks() -> None:
    chunker = FixedSizeChunker(chunk_size=5, chunk_overlap=0)
    embedder = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    service = IngestionService(chunker, embedder, store, batch_size=2)

    documents = [
        Document(id="d1", content="abcdefghij"),  # -> 2 chunks
        Document(id="d2", content="klmno"),  # -> 1 chunk
    ]
    result = service.ingest(documents)

    assert result.documents == 2
    assert result.chunks == 3
    assert len(store) == 3
    assert len(embedder.embedded_texts) == 3


def test_ingest_no_documents() -> None:
    service = IngestionService(FixedSizeChunker(), FakeEmbeddingProvider(), InMemoryVectorStore())
    result = service.ingest([])
    assert result.documents == 0
    assert result.chunks == 0
