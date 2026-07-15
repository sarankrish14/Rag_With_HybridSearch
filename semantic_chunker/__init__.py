"""Semantic PDF chunker — split PDFs into meaning-based chunks with embeddings."""

from semantic_chunker.display import print_results
from semantic_chunker.pipeline import semantic_chunk_pdf
from semantic_chunker.settings import PDF_PATH

__all__ = ["semantic_chunk_pdf", "print_results", "PDF_PATH"]
