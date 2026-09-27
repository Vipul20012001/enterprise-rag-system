from __future__ import annotations

import pytest
from pydantic import ValidationError

from enterprise_rag.config.settings import AppSettings

_REQUIRED = {
    "RAG_AZURE_OPENAI__ENDPOINT": "https://x.openai.azure.com/",
    "RAG_AZURE_OPENAI__API_KEY": "k",
    "RAG_AZURE_OPENAI__CHAT_DEPLOYMENT": "chat",
    "RAG_AZURE_OPENAI__EMBEDDING_DEPLOYMENT": "embed",
    "RAG_AZURE_SEARCH__ENDPOINT": "https://x.search.windows.net",
    "RAG_AZURE_SEARCH__API_KEY": "k2",
}


def test_loads_nested_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in _REQUIRED.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("RAG_RETRIEVAL__TOP_K", "9")

    settings = AppSettings(_env_file=None)

    assert settings.azure_openai.endpoint == "https://x.openai.azure.com/"
    assert settings.azure_openai.chat_deployment == "chat"
    assert settings.azure_search.index_name == "research-papers"  # default
    assert settings.retrieval.top_k == 9
    assert settings.chunking.chunk_size == 1000  # default


def test_missing_required_settings_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _REQUIRED:
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(ValidationError):
        AppSettings(_env_file=None)
