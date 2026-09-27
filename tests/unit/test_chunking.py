from __future__ import annotations

import pytest

from enterprise_rag.application.chunking import FixedSizeChunker
from enterprise_rag.domain.errors import ConfigurationError
from enterprise_rag.domain.models import Document


def test_split_creates_sequential_overlapping_chunks() -> None:
    chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=2)
    document = Document(id="d", content="abcdefghijklmnopqrstuvwxyz")

    chunks = chunker.split(document)

    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert chunks[0].content == "abcdefghij"
    assert chunks[1].content.startswith("ij")  # step = size - overlap = 8
    assert all(chunk.document_id == "d" for chunk in chunks)
    assert all(chunk.id == f"d::{chunk.index}" for chunk in chunks)


def test_split_blank_document_returns_no_chunks() -> None:
    assert FixedSizeChunker().split(Document(id="d", content="   \n\t")) == []


@pytest.mark.parametrize(
    "size, overlap",
    [(0, 0), (5, 5), (5, 6), (5, -1)],
)
def test_invalid_parameters_raise(size: int, overlap: int) -> None:
    with pytest.raises(ConfigurationError):
        FixedSizeChunker(chunk_size=size, chunk_overlap=overlap)
