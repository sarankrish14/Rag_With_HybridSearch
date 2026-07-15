"""
Central configuration for the RagPipelines project.

Non-secret defaults live here. Secrets are loaded via settings.py (Pydantic).
"""

import os
from pathlib import Path

from settings import settings

PROJECT_ROOT: Path = Path(__file__).resolve().parent

# Embedding model
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_BATCH_SIZE: int = 32
EMBEDDING_DIMENSIONS: int = 768

# Vector database (optional ChromaDB mirror)
CHROMA_COLLECTION_NAME: str = "pdf_chunks"
CHROMA_PERSIST_DIR: str = str(PROJECT_ROOT / "chroma_db")

# PostgreSQL — resolved from settings (env / .env only)
DATABASE_URL: str = settings.resolved_database_url

# Hybrid retrieval
RETRIEVAL_VECTOR_TOP_K: int = 20
RETRIEVAL_FTS_TOP_K: int = 20
RETRIEVAL_RRF_K: int = 60
RETRIEVAL_RERANK_TOP_N: int = 5
CROSS_ENCODER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RETRIEVAL_MERGE_STRATEGY: str = "rrf"
RETRIEVAL_VECTOR_WEIGHT: float = 0.6

# Groq LLM — secrets from settings
GROQ_API_KEY: str = settings.groq_api_key
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_TEMPERATURE: float = float(os.getenv("GROQ_TEMPERATURE", "0.2"))
GROQ_MAX_TOKENS: int = int(os.getenv("GROQ_MAX_TOKENS", "1024"))
GROQ_CONTEXT_WINDOW: int = int(os.getenv("GROQ_CONTEXT_WINDOW", "128000"))

# API
API_KEY: str = settings.api_key
API_HOST: str = "0.0.0.0"
API_PORT: int = 8001
