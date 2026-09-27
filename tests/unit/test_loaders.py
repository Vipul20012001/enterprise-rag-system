from __future__ import annotations

from pathlib import Path

import pytest

from enterprise_rag.domain.errors import IngestionError
from enterprise_rag.infrastructure.loaders.text_loader import TextFileLoader


def test_load_directory_filters_by_pattern(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta", encoding="utf-8")
    (tmp_path / "ignore.md").write_text("nope", encoding="utf-8")

    documents = TextFileLoader().load(str(tmp_path))

    assert {document.content for document in documents} == {"alpha", "beta"}
    assert all(document.metadata["filename"].endswith(".txt") for document in documents)


def test_load_single_file(tmp_path: Path) -> None:
    file = tmp_path / "paper.txt"
    file.write_text("content", encoding="utf-8")

    documents = TextFileLoader().load(str(file))

    assert len(documents) == 1
    assert documents[0].id == "paper"
    assert documents[0].content == "content"


def test_missing_path_raises() -> None:
    with pytest.raises(IngestionError):
        TextFileLoader().load("path-that-does-not-exist-xyz")
