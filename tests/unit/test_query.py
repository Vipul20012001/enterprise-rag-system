from __future__ import annotations

import pytest
from fakes import FakeChatModel, FakeEmbeddingProvider, FakeRetriever

from enterprise_rag.application.prompts import DefaultPromptBuilder
from enterprise_rag.application.query import RagQueryService
from enterprise_rag.application.retrieval import VectorRetriever
from enterprise_rag.domain.errors import GenerationError
from enterprise_rag.domain.models import Chunk, EmbeddedChunk, Role
from enterprise_rag.infrastructure.memory.vector_store import InMemoryVectorStore


def test_answer_end_to_end() -> None:
    embedder = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    context = "Paris is the capital of France"
    store.add(
        [
            EmbeddedChunk(
                chunk=Chunk(id="c1", document_id="d", content=context, index=0),
                embedding=embedder.embed_query(context),
            )
        ]
    )
    service = RagQueryService(
        retriever=VectorRetriever(embedder, store),
        chat=(chat := FakeChatModel(reply="Paris")),
        prompt_builder=DefaultPromptBuilder(),
        top_k=3,
    )

    answer = service.answer("What is the capital of France?")

    assert answer.answer == "Paris"
    assert answer.question == "What is the capital of France?"
    assert len(answer.sources) == 1
    assert chat.last_messages is not None
    assert chat.last_messages[0].role == Role.SYSTEM
    assert context in chat.last_messages[1].content


def test_empty_question_raises() -> None:
    service = RagQueryService(FakeRetriever([]), FakeChatModel(), DefaultPromptBuilder())
    with pytest.raises(GenerationError):
        service.answer("   ")


def test_top_k_is_forwarded_to_retriever() -> None:
    retriever = FakeRetriever([])
    service = RagQueryService(retriever, FakeChatModel(), DefaultPromptBuilder(), top_k=7)

    service.answer("hello?")

    assert retriever.calls == [("hello?", 7)]
