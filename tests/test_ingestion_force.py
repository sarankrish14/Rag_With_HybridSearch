"""Tests for force re-ingest (replace-by-hash, no duplicates)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from semantic_chunker.registry import remove_existing_document


@pytest.fixture
def file_hash() -> str:
    return "abc123deadbeef"


def test_remove_existing_document_deletes_postgres_and_registry(file_hash: str) -> None:
    mock_store = MagicMock()
    mock_store.delete_document_by_hash.return_value = "old-doc-id"

    with patch("semantic_chunker.registry.PostgresStore", return_value=mock_store), patch(
        "semantic_chunker.registry.config.DATABASE_URL", "postgresql://localhost/test"
    ), patch("semantic_chunker.registry._load_registry", return_value={file_hash: "old.pdf"}), patch(
        "semantic_chunker.registry._save_registry"
    ) as mock_save, patch(
        "semantic_chunker.storage.delete_chroma_document"
    ) as mock_chroma_delete:
        result = remove_existing_document(file_hash)

    assert result == "old-doc-id"
    mock_store.delete_document_by_hash.assert_called_once_with(file_hash)
    mock_save.assert_called_once()
    saved_registry = mock_save.call_args[0][0]
    assert file_hash not in saved_registry
    mock_chroma_delete.assert_called_once_with("old-doc-id")


def test_remove_existing_document_no_prior_doc(file_hash: str) -> None:
    mock_store = MagicMock()
    mock_store.delete_document_by_hash.return_value = None

    with patch("semantic_chunker.registry.PostgresStore", return_value=mock_store), patch(
        "semantic_chunker.registry.config.DATABASE_URL", "postgresql://localhost/test"
    ), patch("semantic_chunker.registry._load_registry", return_value={}), patch(
        "semantic_chunker.registry._save_registry"
    ) as mock_save, patch("semantic_chunker.storage.delete_chroma_document") as mock_chroma_delete:
        result = remove_existing_document(file_hash)

    assert result is None
    mock_save.assert_not_called()
    mock_chroma_delete.assert_not_called()


def test_delete_document_by_hash_returns_doc_id() -> None:
    from db.postgres_store import PostgresStore

    store = PostgresStore.__new__(PostgresStore)
    store.get_document_by_hash = MagicMock(return_value={"doc_id": "doc-99"})
    store.delete_document = MagicMock()

    result = PostgresStore.delete_document_by_hash(store, "hash-1")

    assert result == "doc-99"
    store.delete_document.assert_called_once_with("doc-99")
