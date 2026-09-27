"""Default chunking strategy: fixed-size character windows with overlap.

Deliberately simple for the Phase 1 baseline. Smarter strategies (sentence-,
token-, or layout-aware) are added later by implementing
:class:`enterprise_rag.domain.ports.ChunkingStrategy`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..domain.errors import ConfigurationError
from ..domain.models import Chunk, Document

__all__ = ["FixedSizeChunker"]


@dataclass(frozen=True)
class FixedSizeChunker:
    """Split text into ``chunk_size`` windows overlapping by ``chunk_overlap``."""

    chunk_size: int = 1000
    chunk_overlap: int = 200

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ConfigurationError("chunk_size must be a positive integer")
        if self.chunk_overlap < 0:
            raise ConfigurationError("chunk_overlap must not be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigurationError("chunk_overlap must be smaller than chunk_size")

    def split(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text.strip():
            return []

        step = self.chunk_size - self.chunk_overlap
        chunks: list[Chunk] = []
        start = 0
        index = 0
        while start < len(text):
            window = text[start : start + self.chunk_size]
            if window.strip():
                chunks.append(
                    Chunk(
                        id=f"{document.id}::{index}",
                        document_id=document.id,
                        content=window,
                        index=index,
                        metadata=document.metadata,
                    )
                )
                index += 1
            start += step
        return chunks
