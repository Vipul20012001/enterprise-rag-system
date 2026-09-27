"""Infrastructure layer: concrete adapters implementing the domain ports.

Submodules are imported explicitly (e.g. ``from enterprise_rag.infrastructure.azure_search
import AzureAISearchVectorStore``) so that importing one adapter family does not
force optional SDK imports from another.
"""

from __future__ import annotations
