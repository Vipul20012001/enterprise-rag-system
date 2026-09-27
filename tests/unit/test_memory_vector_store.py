from __future__ import annotations

from enterprise_rag.domain.models import Chunk, EmbeddedChunk
from enterprise_rag.infrastructure.memory.vector_store import InMemoryVectorStore


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(id=chunk_id, document_id="d", content=chunk_id, index=0)


def test_search_orders_by_cosine_similarity() -> None:
    store = InMemoryVectorStore()
    store.add(
        [
            EmbeddedChunk(chunk=_chunk("a"), embedding=[1.0, 0.0]),
            EmbeddedChunk(chunk=_chunk("b"), embedding=[0.0, 1.0]),
        ]
    )

    results = store.search([1.0, 0.0], top_k=2)

    assert [result.chunk.id for result in results] == ["a", "b"]
    assert results[0].score > results[1].score


def test_top_k_limits_results() -> None:
    store = InMemoryVectorStore()
    store.add([EmbeddedChunk(chunk=_chunk(str(i)), embedding=[float(i)]) for i in range(1, 6)])

    assert len(store.search([1.0], top_k=3)) == 3


def test_empty_store_returns_empty() -> None:
    assert InMemoryVectorStore().search([1.0], top_k=3) == []
