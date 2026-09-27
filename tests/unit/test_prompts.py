from __future__ import annotations

from enterprise_rag.application.prompts import SYSTEM_PROMPT, DefaultPromptBuilder
from enterprise_rag.domain.models import Chunk, RetrievedChunk, Role


def test_build_includes_system_prompt_and_context() -> None:
    builder = DefaultPromptBuilder()
    contexts = [
        RetrievedChunk(
            chunk=Chunk(id="c1", document_id="d", content="ctx text", index=0),
            score=0.9,
        )
    ]

    messages = builder.build("What?", contexts)

    assert messages[0].role == Role.SYSTEM
    assert messages[0].content == SYSTEM_PROMPT
    assert messages[1].role == Role.USER
    assert "[c1] ctx text" in messages[1].content
    assert "What?" in messages[1].content


def test_build_without_context() -> None:
    messages = DefaultPromptBuilder().build("q", [])
    assert "No context available." in messages[1].content
