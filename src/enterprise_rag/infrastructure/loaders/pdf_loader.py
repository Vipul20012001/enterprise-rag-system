"""Loader that extracts text from PDF research papers using ``pypdf``."""

from __future__ import annotations

from pathlib import Path

from ...domain.errors import IngestionError
from ...domain.models import Document

__all__ = ["PdfLoader"]


class PdfLoader:
    """Load ``*.pdf`` files (a file or a directory) into :class:`Document` objects."""

    def __init__(self, *, pattern: str = "*.pdf") -> None:
        self._pattern = pattern

    def load(self, source: str) -> list[Document]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - depends on optional runtime dep
            raise IngestionError("pypdf is required to load PDF files") from exc

        path = Path(source)
        if not path.exists():
            raise IngestionError(f"source path does not exist: {source}")

        files = [path] if path.is_file() else sorted(path.rglob(self._pattern))
        documents: list[Document] = []
        for file in files:
            reader = PdfReader(str(file))
            pages = [page.extract_text() or "" for page in reader.pages]
            content = "\n".join(pages)
            documents.append(
                Document(
                    id=file.stem or str(file),
                    content=content,
                    source=str(file),
                    title=file.stem,
                    metadata={"filename": file.name, "pages": str(len(reader.pages))},
                )
            )
        return documents
