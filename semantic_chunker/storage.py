"""PostgreSQL and ChromaDB persistence."""

from pathlib import Path
from typing import Any

import numpy as np

import config
from db.postgres_store import PostgresStore
from semantic_chunker.logging_setup import logger


def save_to_postgres(
  doc_id: str,
  pdf_path: str,
  file_hash: str,
  page_map: list[dict],
  chunks: list[dict],
  chunk_embeddings: np.ndarray,
) -> None:
  if not config.DATABASE_URL:
    logger.warning("DATABASE_URL not set — skipping Postgres persistence")
    return

  store = PostgresStore()
  store.init_db()

  page_numbers = [p["page_no"] for p in page_map if p.get("page_no") is not None]
  page_count = max(page_numbers) if page_numbers else None

  docling_metadata = {
    "element_count": len(page_map),
    "chunk_count": len(chunks),
  }

  store.insert_document(
    doc_id=doc_id,
    file_name=Path(pdf_path).name,
    file_hash=file_hash,
    file_path=pdf_path,
    page_count=page_count,
    docling_metadata=docling_metadata,
  )

  postgres_chunks = [
    {
      **chunk,
      "doc_id": doc_id,
      "embedding": chunk_embeddings[i].tolist(),
    }
    for i, chunk in enumerate(chunks)
  ]

  inserted = store.insert_chunks(doc_id, postgres_chunks)
  logger.info(
    f"Stored {inserted} chunk embeddings in PostgreSQL (doc_id={doc_id})"
  )


_chroma_collection: Any | None = None


def _get_chroma_collection() -> Any | None:
  global _chroma_collection
  if _chroma_collection is not None:
    return _chroma_collection

  try:
    from chromastore import ChromaStore

    store = ChromaStore(persist_dir=config.CHROMA_PERSIST_DIR)
    _chroma_collection = store.collection
    return _chroma_collection
  except Exception as e:
    logger.warning(f"ChromaDB unavailable — skipping vector store: {e}")
    return None


def save_to_chroma(
  collection: Any,
  chunks: list[dict],
  chunk_embeddings: np.ndarray,
) -> None:
  if not chunks:
    logger.warning("No chunks to save to ChromaDB")
    return

  ids: list[str] = []
  documents: list[str] = []
  embeddings: list[list[float]] = []
  metadatas: list[dict] = []

  for i, chunk in enumerate(chunks):
    ids.append(chunk["chunk_id"])
    documents.append(chunk["text"])
    embeddings.append(chunk_embeddings[i].tolist())
    doc_id_val = chunk.get("doc_id", "")
    if not doc_id_val:
      logger.warning(
        f"Chunk '{chunk['chunk_id']}' has no doc_id set — "
        f"ChromaDB metadata will store empty string. "
        f"Ensure doc_id is injected before calling save_to_chroma()."
      )
    metadatas.append({
      "doc_id": doc_id_val,
      "chunk_index": chunk["chunk_index"],
      "sentence_count": chunk["sentence_count"],
      "char_count": chunk["char_count"],
      "source": chunk["source"],
      "file_name": chunk.get("file_name", ""),
      "page_no": chunk["page_no"] if chunk["page_no"] is not None else -1,
      "chunk_type": chunk.get("chunk_type", "text"),
      "threshold_used": chunk["threshold_used"],
    })

  collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
  )

  logger.info(f"Saved {len(ids)} chunks to ChromaDB")


def delete_chroma_document(doc_id: str) -> None:
  """Remove all Chroma chunks belonging to a document."""
  try:
    from chromastore import ChromaStore

    store = ChromaStore(persist_dir=config.CHROMA_PERSIST_DIR)
    store.delete_document(doc_id)
    logger.info(f"Removed Chroma chunks for doc_id={doc_id}")
  except Exception as e:
    logger.warning(f"Chroma delete failed for doc_id={doc_id}: {e}")
