"""Sentence and chunk embedding with MPNet."""

import numpy as np
from sentence_transformers import SentenceTransformer

from semantic_chunker.logging_setup import logger
from semantic_chunker.settings import EMBED_BATCH_SIZE, EMBED_DIMENSION

_model_cache: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
  global _model_cache
  if _model_cache is None:
    logger.info("🤖 Loading model: sentence-transformers/all-mpnet-base-v2")
    _model_cache = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
  return _model_cache


def embed_sentences(sentences: list[str], model: SentenceTransformer) -> np.ndarray:
  all_embeddings = []
  for batch_start in range(0, len(sentences), EMBED_BATCH_SIZE):
    batch = sentences[batch_start : batch_start + EMBED_BATCH_SIZE]
    try:
      batch_embeddings = model.encode(
        batch,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
      )
      all_embeddings.append(batch_embeddings)
    except Exception as e:
      logger.error(
        f"Embedding failed on batch {batch_start}–"
        f"{batch_start + len(batch) - 1}: {e} — skipping batch"
      )
      all_embeddings.append(
        np.zeros((len(batch), EMBED_DIMENSION), dtype=np.float32)
      )
  if not all_embeddings:
    logger.error("Embedding failed: no input sentences")
    raise RuntimeError("Embedding failed: no input sentences")
  return np.vstack(all_embeddings)


def embed_chunks(chunks: list[str], model: SentenceTransformer) -> np.ndarray:
  all_embeddings = []
  for batch_start in range(0, len(chunks), EMBED_BATCH_SIZE):
    batch = chunks[batch_start : batch_start + EMBED_BATCH_SIZE]
    try:
      batch_embeddings = model.encode(
        batch,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
      )
      all_embeddings.append(batch_embeddings)
    except Exception as e:
      logger.error(
        f"Embedding failed on batch {batch_start}–"
        f"{batch_start + len(batch) - 1}: {e} — skipping batch"
      )
      all_embeddings.append(
        np.zeros((len(batch), EMBED_DIMENSION), dtype=np.float32)
      )
  if not all_embeddings:
    logger.error("Chunk embedding failed: no input chunks")
    raise RuntimeError("Chunk embedding failed: no input chunks")
  return np.vstack(all_embeddings)
