"""Loader that reads plain-text files from a file or directory."""

from __future__ import annotations

from pathlib import Path

from ...domain.errors import IngestionError
from ...domain.models import Document

__all__ = ["TextFileLoader"]


class TextFileLoader:
    """Load ``*.txt`` (or a custom glob) files into :class:`Document` objects."""

    def __init__(self, *, encoding: str = "utf-8", pattern: str = "*.txt") -> None:
        self._encoding = encoding
        self._pattern = pattern

    def load(self, source: str) -> list[Document]:
        path = Path(source)
        if not path.exists():
            raise IngestionError(f"source path does not exist: {source}")

        files = [path] if path.is_file() else sorted(path.rglob(self._pattern))
        documents: list[Document] = []
        for file in files:
            content = file.read_text(encoding=self._encoding)
            documents.append(
                Document(
                    id=file.stem or str(file),
                    content=content,
                    source=str(file),
                    title=file.stem,
                    metadata={"filename": file.name},
                )
            )
        return documents
