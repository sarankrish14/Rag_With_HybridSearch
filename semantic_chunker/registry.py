"""Duplicate detection and seen-PDF registry."""

import hashlib
import json
from pathlib import Path

import config
from db.postgres_store import PostgresStore
from semantic_chunker.logging_setup import logger
from semantic_chunker.settings import SEEN_REGISTRY


def _hash_file(path: str) -> str:
  h = hashlib.md5()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(8192), b""):
      h.update(chunk)
  return h.hexdigest()


def _load_registry() -> dict:
  if Path(SEEN_REGISTRY).exists():
    with open(SEEN_REGISTRY, "r") as f:
      return json.load(f)
  return {}


def _save_registry(registry: dict) -> None:
  Path(SEEN_REGISTRY).parent.mkdir(parents=True, exist_ok=True)
  with open(SEEN_REGISTRY, "w") as f:
    json.dump(registry, f, indent=2)


def is_duplicate(pdf_path: str) -> bool:
  file_hash = _hash_file(pdf_path)

  if config.DATABASE_URL:
    try:
      logger.info("Checking Postgres for duplicate document...")
      store = PostgresStore()
      existing = store.get_document_by_hash(file_hash)
      if existing:
        logger.warning(
          f"Duplicate detected in Postgres: '{pdf_path}' "
          f"→ doc_id={existing['doc_id']}"
        )
        return True
    except Exception as e:
      logger.error(f"Postgres duplicate check failed: {e}")

  registry = _load_registry()
  if file_hash in registry:
    logger.warning(
      f"Duplicate detected: '{pdf_path}' was already processed as '{registry[file_hash]}'"
    )
    return True

  return False


def mark_as_processed(pdf_path: str, file_hash: str | None = None) -> None:
  if file_hash is None:
    file_hash = _hash_file(pdf_path)
  registry = _load_registry()
  registry[file_hash] = pdf_path
  _save_registry(registry)
  logger.debug(f"Registered file hash {file_hash} → {pdf_path}")


def remove_existing_document(file_hash: str) -> str | None:
  """
  Remove a previously ingested document before force re-ingest.

  Deletes PostgreSQL rows (chunks cascade) and clears the local registry entry.
  Returns the removed doc_id, or None if nothing existed.
  """
  old_doc_id: str | None = None

  if config.DATABASE_URL:
    try:
      store = PostgresStore()
      old_doc_id = store.delete_document_by_hash(file_hash)
      if old_doc_id:
        logger.info(
          "Force re-ingest: removed existing document doc_id=%s (hash=%s)",
          old_doc_id,
          file_hash,
        )
    except Exception as e:
      logger.error(f"Failed to remove existing document by hash: {e}")
      raise

  registry = _load_registry()
  if file_hash in registry:
    del registry[file_hash]
    _save_registry(registry)

  if old_doc_id:
    from semantic_chunker.storage import delete_chroma_document

    delete_chroma_document(old_doc_id)

  return old_doc_id
