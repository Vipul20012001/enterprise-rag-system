"""Azure OpenAI chat completion adapter.

Wraps the official ``openai`` SDK's ``AzureOpenAI`` client behind the
:class:`enterprise_rag.domain.ports.ChatModel` port.
"""

from __future__ import annotations

from collections.abc import Sequence

from openai import AzureOpenAI

from ...domain.errors import GenerationError
from ...domain.models import ChatMessage

__all__ = ["AzureOpenAIChatModel"]


class AzureOpenAIChatModel:
    """Generates answers using an Azure OpenAI chat deployment."""

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        api_version: str,
        deployment: str,
        temperature: float = 0.0,
    ) -> None:
        self._client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        self._deployment = deployment
        self._temperature = temperature

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        payload = [{"role": message.role.value, "content": message.content} for message in messages]
        response = self._client.chat.completions.create(
            model=self._deployment,
            messages=payload,  # type: ignore[arg-type]
            temperature=self._temperature,
        )
        content = response.choices[0].message.content
        if content is None:
            raise GenerationError("Azure OpenAI returned an empty completion")
        return content
