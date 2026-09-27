from __future__ import annotations

from enterprise_rag.infrastructure.memory.embeddings import HashingEmbeddingProvider


def test_embedding_is_deterministic() -> None:
    provider = HashingEmbeddingProvider(dimensions=32)
    assert provider.embed_query("hello world") == provider.embed_query("hello world")


def test_embedding_has_requested_dimensions() -> None:
    provider = HashingEmbeddingProvider(dimensions=64)
    assert len(provider.embed_query("hello")) == 64


def test_embed_texts_returns_one_vector_per_input() -> None:
    provider = HashingEmbeddingProvider(dimensions=8)
    vectors = provider.embed_texts(["a b c", "d"])
    assert len(vectors) == 2
    assert all(len(vector) == 8 for vector in vectors)
