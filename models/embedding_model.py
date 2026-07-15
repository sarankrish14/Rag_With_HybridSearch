"""Singleton loader for the local sentence-transformers embedding model."""

from __future__ import annotations

import logging
import threading
from typing import ClassVar

import numpy as np
from sentence_transformers import SentenceTransformer

import config

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Thread-safe singleton wrapping SentenceTransformer."""

    _instance: ClassVar["EmbeddingModel | None"] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls) -> "EmbeddingModel":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        with self._lock:
            if self._initialized:
                return
            logger.info("Loading embedding model: %s", config.EMBEDDING_MODEL_NAME)
            self._model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
            self._initialized = True
            logger.info("Embedding model ready (%d dimensions)", config.EMBEDDING_DIMENSIONS)

    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        if not texts:
            raise ValueError("encode() requires at least one text")
        effective_batch = batch_size or config.EMBEDDING_BATCH_SIZE
        embeddings: np.ndarray = self._model.encode(
            texts,
            batch_size=effective_batch,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings


def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()
