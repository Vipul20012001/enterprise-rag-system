"""Azure AI Search vector store adapter.

Wraps ``azure-search-documents`` behind the
:class:`enterprise_rag.domain.ports.VectorStore` port. Azure SDK types are
confined to this module.

Document keys in Azure AI Search may only contain letters, digits, ``_``, ``-``
and ``=``; chunk ids may not, so the original id is URL-safe base64 encoded for
the ``id`` key field and preserved verbatim in ``chunk_id``.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Sequence

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery

from ...domain.models import Chunk, EmbeddedChunk, RetrievedChunk

__all__ = ["AzureAISearchVectorStore"]

_VECTOR_FIELD = "embedding"
_ALGORITHM_NAME = "hnsw-default"
_PROFILE_NAME = "vector-profile"


class AzureAISearchVectorStore:
    """Persist and retrieve embedded chunks in an Azure AI Search index."""

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        index_name: str,
        vector_dimensions: int = 1536,
    ) -> None:
        self._endpoint = endpoint
        self._credential = AzureKeyCredential(api_key)
        self._index_name = index_name
        self._dimensions = vector_dimensions
        self._client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=self._credential,
        )

    def ensure_index(self) -> None:
        """Create (or update) the target index with a vector field."""
        index_client = SearchIndexClient(endpoint=self._endpoint, credential=self._credential)
        index = SearchIndex(
            name=self._index_name,
            fields=[
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SimpleField(name="chunk_id", type=SearchFieldDataType.String),
                SimpleField(name="document_id", type=SearchFieldDataType.String, filterable=True),
                SearchableField(name="content", type=SearchFieldDataType.String),
                SimpleField(name="metadata", type=SearchFieldDataType.String),
                SearchField(
                    name=_VECTOR_FIELD,
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=self._dimensions,
                    vector_search_profile_name=_PROFILE_NAME,
                ),
            ],
            vector_search=VectorSearch(
                algorithms=[HnswAlgorithmConfiguration(name=_ALGORITHM_NAME)],
                profiles=[
                    VectorSearchProfile(
                        name=_PROFILE_NAME,
                        algorithm_configuration_name=_ALGORITHM_NAME,
                    )
                ],
            ),
        )
        index_client.create_or_update_index(index)

    def add(self, chunks: Sequence[EmbeddedChunk]) -> None:
        if not chunks:
            return
        documents = [
            {
                "id": _encode_key(item.chunk.id),
                "chunk_id": item.chunk.id,
                "document_id": item.chunk.document_id,
                "content": item.chunk.content,
                "metadata": json.dumps(dict(item.chunk.metadata)),
                _VECTOR_FIELD: list(item.embedding),
            }
            for item in chunks
        ]
        self._client.upload_documents(documents=documents)

    def search(self, embedding: Sequence[float], top_k: int) -> list[RetrievedChunk]:
        if top_k <= 0:
            return []
        vector_query = VectorizedQuery(
            vector=list(embedding),
            k_nearest_neighbors=top_k,
            fields=_VECTOR_FIELD,
        )
        results = self._client.search(
            search_text=None,
            vector_queries=[vector_query],
            top=top_k,
        )

        retrieved: list[RetrievedChunk] = []
        for result in results:
            metadata_raw = result.get("metadata") or "{}"
            chunk = Chunk(
                id=result.get("chunk_id") or result["id"],
                document_id=result.get("document_id", ""),
                content=result.get("content", ""),
                index=0,
                metadata=json.loads(metadata_raw),
            )
            retrieved.append(
                RetrievedChunk(chunk=chunk, score=float(result.get("@search.score", 0.0)))
            )
        return retrieved


def _encode_key(chunk_id: str) -> str:
    return base64.urlsafe_b64encode(chunk_id.encode("utf-8")).decode("ascii").rstrip("=")
