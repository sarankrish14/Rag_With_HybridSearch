"""Main semantic PDF chunking pipeline."""

import uuid
from pathlib import Path
from typing import Any

import config
from db.postgres_store import PostgresStore
from semantic_chunker.chunking import (
  build_chunks,
  find_breakpoints,
  is_garbage_chunk,
)
from semantic_chunker.embedding import _get_model, embed_chunks, embed_sentences
from semantic_chunker.extract import extract_text_from_pdf
from semantic_chunker.logging_setup import logger
from semantic_chunker.registry import (
  _hash_file,
  is_duplicate,
  mark_as_processed,
  remove_existing_document,
)
from semantic_chunker.sentences import split_into_sentences
from semantic_chunker.storage import (
  _get_chroma_collection,
  save_to_chroma,
  save_to_postgres,
)


def semantic_chunk_pdf(
  pdf_path: str,
  force: bool = False,
  chroma_collection: Any | None = None,
) -> list[dict]:
  if config.DATABASE_URL:
    try:
      PostgresStore().init_db()
    except Exception as e:
      logger.error(f"Postgres schema init failed: {e}")

  if not force and is_duplicate(pdf_path):
    logger.warning("Skipping — already processed. Pass force=True to re-process.")
    return []

  file_hash = _hash_file(pdf_path)

  if force:
    removed_doc_id = remove_existing_document(file_hash)
    if removed_doc_id:
      logger.info(
        "Force re-ingest: replaced prior document doc_id=%s for '%s'",
        removed_doc_id,
        pdf_path,
      )

  try:
    logger.info(f"📄 Step 1 — Extracting text from: {pdf_path}")
    _full_text, page_map = extract_text_from_pdf(pdf_path)

    text_items = [item for item in page_map if item.get("type") == "text"]
    table_items = [item for item in page_map if item.get("type") == "table"]
    logger.info(
      f"Found {len(text_items)} text elements and {len(table_items)} table elements"
    )

    logger.info("✂️  Step 2 — Splitting text into sentences (tables skipped)...")
    sentences: list[str] = []
    sentence_pages: list[int | None] = []

    for item in page_map:
      if item.get("type") == "table":
        continue
      for s in split_into_sentences(item["text"]):
        sentences.append(s)
        sentence_pages.append(item["page_no"])

    logger.info(f"Found {len(sentences)} sentences from text elements")

    logger.info("🤖 Step 3 — Getting embedding model...")
    model = _get_model()

    results: list[dict] = []
    threshold_used = 0.0

    if sentences:
      logger.info("🔢 Step 4 — Embedding sentences...")
      embeddings = embed_sentences(sentences, model)
      logger.info(f"Embedding shape: {embeddings.shape}")

      logger.info("🔍 Step 5 — Detecting semantic breakpoints...")
      short_doc_threshold = 10
      if len(sentences) < short_doc_threshold:
        logger.warning(
          f"Short document — only {len(sentences)} sentences found. "
          f"Skipping breakpoint detection and treating entire text as one chunk."
        )
        breakpoints = [0]
        threshold_used = 0.0
      else:
        breakpoints, threshold_used = find_breakpoints(embeddings)
      logger.info(
        f"Found {len(breakpoints)} breakpoints at threshold {threshold_used:.3f}"
      )

      logger.info("📦 Step 6 — Building semantic text chunks...")
      chunk_sentences, original_lengths, start_offsets = build_chunks(
        sentences, breakpoints
      )

      for chunk_idx, sents in enumerate(chunk_sentences):
        chunk_text = " ".join(sents)

        if is_garbage_chunk(chunk_text):
          logger.debug(f"Skipping garbage text chunk: '{chunk_text[:60]}...'")
          continue

        offset = start_offsets[chunk_idx]
        page_no = (
          sentence_pages[offset]
          if offset < len(sentence_pages)
          else None
        )

        results.append({
          "chunk_id": str(uuid.uuid4()),
          "chunk_index": 0,
          "text": chunk_text,
          "sentence_count": original_lengths[chunk_idx],
          "char_count": len(chunk_text),
          "source": pdf_path,
          "file_name": Path(pdf_path).name,
          "page_no": page_no,
          "chunk_type": "text",
          "threshold_used": round(threshold_used, 4),
        })
    else:
      logger.info("No text sentences found — skipping semantic text chunking")

    logger.info("📊 Step 6b — Building table chunks (no splitting)...")
    table_chunks: list[dict] = []
    for item in table_items:
      if is_garbage_chunk(item["text"]):
        logger.debug(f"Skipping garbage table chunk: '{item['text'][:60]}...'")
        continue
      table_chunks.append({
        "chunk_id": str(uuid.uuid4()),
        "chunk_index": 0,
        "text": item["text"],
        "sentence_count": 1,
        "char_count": len(item["text"]),
        "source": pdf_path,
        "file_name": Path(pdf_path).name,
        "page_no": item.get("page_no"),
        "chunk_type": "table",
        "threshold_used": 0.0,
      })
      if item.get("page_no") is None:
        logger.debug(
          f"Table chunk '{table_chunks[-1]['chunk_id']}' has no page_no "
          f"(Docling could not determine page — prov was empty)"
        )

    all_chunks = results + table_chunks
    for i, chunk in enumerate(all_chunks):
      chunk["chunk_index"] = i + 1
      chunk.setdefault("chunk_type", "text")

    logger.info(
      f"Built {len(results)} text chunks + {len(table_chunks)} table chunks "
      f"= {len(all_chunks)} total"
    )

    if all_chunks:
      logger.info("🔢 Step 7 — Embedding final chunks for PostgreSQL storage...")
      chunk_texts = [c["text"] for c in all_chunks]
      chunk_embeddings = embed_chunks(chunk_texts, model)

      doc_id = str(uuid.uuid4())

      for chunk in all_chunks:
        chunk["doc_id"] = doc_id

      logger.info("💾 Step 8 — Saving to PostgreSQL...")
      save_to_postgres(
        doc_id, pdf_path, file_hash, page_map, all_chunks, chunk_embeddings
      )

      chroma = (
        chroma_collection
        if chroma_collection is not None
        else _get_chroma_collection()
      )
      if chroma is not None:
        logger.info("💾 Step 9 — Saving to ChromaDB...")
        save_to_chroma(chroma, all_chunks, chunk_embeddings)

    mark_as_processed(pdf_path, file_hash=file_hash)
    logger.info(f"✅ Done — {len(all_chunks)} chunks created from '{pdf_path}'")

    return all_chunks

  except Exception as e:
    logger.error(f"Pipeline failed for '{pdf_path}': {e}")
    raise
