"""Azure OpenAI embedding adapter.

Wraps the official ``openai`` SDK's ``AzureOpenAI`` client. The SDK type never
leaves this module: callers depend on the
:class:`enterprise_rag.domain.ports.EmbeddingProvider` port.
"""

from __future__ import annotations

from collections.abc import Sequence

from openai import AzureOpenAI

__all__ = ["AzureOpenAIEmbeddingProvider"]


class AzureOpenAIEmbeddingProvider:
    """Embeds text using an Azure OpenAI embedding deployment."""

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        api_version: str,
        deployment: str,
    ) -> None:
        self._client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        self._deployment = deployment

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self._deployment, input=list(texts))
        return [list(item.embedding) for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]
