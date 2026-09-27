"""Default prompt builder for the Naive RAG baseline."""

from __future__ import annotations

from collections.abc import Sequence

from ..domain.models import ChatMessage, RetrievedChunk, Role

__all__ = ["DefaultPromptBuilder", "SYSTEM_PROMPT"]

SYSTEM_PROMPT = (
    "You are a research assistant answering questions about scientific papers. "
    "Answer the user's question using only the provided context. "
    "If the context does not contain the answer, say that you do not know. "
    "Cite the sources you used by their bracketed identifiers."
)


class DefaultPromptBuilder:
    """Builds a system + user message pair with the retrieved context inlined."""

    def build(self, question: str, contexts: Sequence[RetrievedChunk]) -> list[ChatMessage]:
        if contexts:
            context_block = "\n\n".join(
                f"[{item.chunk.id}] {item.chunk.content}" for item in contexts
            )
        else:
            context_block = "No context available."

        user_content = f"Context:\n{context_block}\n\nQuestion: {question}"
        return [
            ChatMessage(role=Role.SYSTEM, content=SYSTEM_PROMPT),
            ChatMessage(role=Role.USER, content=user_content),
        ]
