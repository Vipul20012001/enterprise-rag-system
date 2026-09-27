"""Query use case: retrieve context, build a prompt, generate an answer."""

from __future__ import annotations

from ..domain.errors import GenerationError
from ..domain.models import RagAnswer
from ..domain.ports import ChatModel, PromptBuilder, Retriever

__all__ = ["RagQueryService"]


class RagQueryService:
    """Naive RAG: retrieve -> prompt -> generate."""

    def __init__(
        self,
        retriever: Retriever,
        chat: ChatModel,
        prompt_builder: PromptBuilder,
        *,
        top_k: int = 5,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        self._retriever = retriever
        self._chat = chat
        self._prompt_builder = prompt_builder
        self._top_k = top_k

    def answer(self, question: str) -> RagAnswer:
        if not question.strip():
            raise GenerationError("question must not be empty")

        contexts = self._retriever.retrieve(question, self._top_k)
        messages = self._prompt_builder.build(question, contexts)
        answer_text = self._chat.complete(messages)
        return RagAnswer(question=question, answer=answer_text, sources=tuple(contexts))
