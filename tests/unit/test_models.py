from __future__ import annotations

import dataclasses

import pytest

from enterprise_rag.domain.models import ChatMessage, Document, RagAnswer, Role


def test_document_defaults() -> None:
    document = Document(id="d1", content="hello")
    assert document.source == ""
    assert document.title is None
    assert document.metadata == {}


def test_models_are_frozen() -> None:
    message = ChatMessage(role=Role.USER, content="hi")
    with pytest.raises(dataclasses.FrozenInstanceError):
        message.content = "changed"  # type: ignore[misc]


def test_rag_answer_default_sources() -> None:
    answer = RagAnswer(question="q", answer="a")
    assert answer.sources == ()


def test_role_is_string_enum() -> None:
    assert Role.SYSTEM.value == "system"
    assert Role.USER == "user"
