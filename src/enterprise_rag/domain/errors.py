"""Domain error hierarchy."""

from __future__ import annotations

__all__ = [
    "RagError",
    "ConfigurationError",
    "IngestionError",
    "RetrievalError",
    "GenerationError",
]


class RagError(Exception):
    """Base class for all errors raised by this package."""


class ConfigurationError(RagError):
    """Raised when configuration is missing or invalid."""


class IngestionError(RagError):
    """Raised when documents cannot be ingested."""


class RetrievalError(RagError):
    """Raised when retrieval fails."""


class GenerationError(RagError):
    """Raised when answer generation fails."""
