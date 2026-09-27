"""Document loader adapters."""

from __future__ import annotations

from .pdf_loader import PdfLoader
from .text_loader import TextFileLoader

__all__ = ["TextFileLoader", "PdfLoader"]
