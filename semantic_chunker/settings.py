"""Configuration for the semantic PDF chunker."""

from config import PROJECT_ROOT

PDF_PATH = "documents/sample.pdf"
MIN_CHUNK_SIZE = 2
MAX_CHUNK_SIZE = 20
OVERLAP_SENTENCES = 3
EMBED_BATCH_SIZE = 64
MIN_CHUNK_CHARS = 100
SEEN_REGISTRY = str(PROJECT_ROOT / "semantic_chunker" / "seen_pdfs.json")
LOG_FILE = str(PROJECT_ROOT / "semantic_chunker" / "chunker.log")
CHROMA_COLLECTION_NAME = "pdf_chunks"
EMBED_DIMENSION = 768
