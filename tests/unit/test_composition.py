from __future__ import annotations

import pytest

pytest.importorskip("openai")
pytest.importorskip("azure.search.documents")

from enterprise_rag.composition import RagServices, build_services  # noqa: E402
from enterprise_rag.config.settings import (  # noqa: E402
    AppSettings,
    AzureOpenAISettings,
    AzureSearchSettings,
)


def _settings() -> AppSettings:
    return AppSettings(
        _env_file=None,
        azure_openai=AzureOpenAISettings(
            endpoint="https://x.openai.azure.com/",
            api_key="k",
            chat_deployment="chat",
            embedding_deployment="embed",
        ),
        azure_search=AzureSearchSettings(
            endpoint="https://x.search.windows.net",
            api_key="k2",
        ),
    )


def test_build_services_wires_pipeline_offline() -> None:
    services = build_services(_settings())

    assert isinstance(services, RagServices)
    # The Azure store exposes index provisioning used by the `init-index` command.
    assert hasattr(services.store, "ensure_index")
