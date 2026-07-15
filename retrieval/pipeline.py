"""
retrieval/pipeline.py — Full production RAG retrieval pipeline (Steps 1–11).

WHAT THIS FILE DOES:
    End-to-end retrieval: raw user query → cleaned query → embedding → hybrid
    search (PostgreSQL pgvector + BM25) → metadata filter → cross-encoder
    rerank → neighbor context expansion → LLM prompt → Groq generation →
    citation validation → formatted answer.

STORAGE BACKEND:
    Reads chunks from PostgreSQL (pgvector for dense, in-memory BM25 for sparse).
    This matches the ingestion pipeline that persists via db.postgres_store.

INPUT:
    Raw query string (+ optional metadata filters) from API or CLI.

OUTPUT:
    PipelineResult with answer, citations, latency breakdown, and diagnostics.

UPSTREAM:
    db.postgres_store.PostgresStore, models.embedding_model, project config.py

DOWNSTREAM:
    FastAPI endpoints in this file (app), or direct RAGPipeline.run() calls.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import logging
import math
import os
import pickle
import re
import sys
import threading
import time
import unicodedata
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

import config as project_config
from db.async_postgres_store import AsyncPostgresStore
from db.postgres_store import PostgresStore
from models.embedding_model import get_embedding_model
from retrieval.groq_async import AsyncGroqClient, GroqAPIError, SyncGroqClient
from retrieval.token_budget import (
    count_messages_tokens,
    fit_prompt_to_budget,
    max_input_tokens,
    trim_chunks_to_budget,
)

# ---------------------------------------------------------------------------
# Optional dependencies — degrade gracefully when not installed.
# ---------------------------------------------------------------------------
try:
    from langdetect import LangDetectException, detect as langdetect_detect

    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False

try:
    import redis

    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False

try:
    from prometheus_client import Histogram

    _PIPELINE_HIST = Histogram(
        "rag_step_duration_ms",
        "Pipeline step latency",
        labelnames=["step"],
    )
except ImportError:
    _PIPELINE_HIST = None

try:
    import pybreaker

    _PYBREAKER_AVAILABLE = True
except ImportError:
    _PYBREAKER_AVAILABLE = False

try:
    import jsonschema

    _JSONSCHEMA_AVAILABLE = True
except ImportError:
    _JSONSCHEMA_AVAILABLE = False

try:
    from fastapi import Depends, FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address
    from pydantic import BaseModel, Field, field_validator
    import uvicorn

    from retrieval.auth import verify_api_key

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Logging — one shared format for every step in this pipeline.
# ---------------------------------------------------------------------------
_REQUEST_ID_CTX: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)
_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | req=%(request_id)s | %(message)s"


class _RequestIDFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _REQUEST_ID_CTX.get("-")
        return True


_REQUEST_ID_FILTER = _RequestIDFilter()


def _get_logger(name: str) -> logging.Logger:
    """Return a module logger under the rag.retrieval.* namespace."""
    logger = logging.getLogger(f"rag.retrieval.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        handler.addFilter(_REQUEST_ID_FILTER)
        logger.addHandler(handler)
    return logger


def _ensure_request_id() -> str:
    current = _REQUEST_ID_CTX.get()
    if current:
        return current
    request_id = str(uuid.uuid4())[:8]
    _REQUEST_ID_CTX.set(request_id)
    return request_id


def _observe_pipeline_metrics(breakdown: dict[str, float]) -> None:
    if _PIPELINE_HIST is None:
        return
    for step, ms in breakdown.items():
        _PIPELINE_HIST.labels(step=step).observe(ms)


_JSON_PARSE_FAILURES = 0
_JSON_PARSE_LOCK = threading.Lock()

LLM_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["answer", "citations"],
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array"},
    },
}


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ms_since(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


# =============================================================================
# CONFIG — all tunables in one place; env RAG_* overrides supported.
# =============================================================================


@dataclass
class _Secret:
    _value: str

    def __repr__(self) -> str:
        return "***"

    def __str__(self) -> str:
        return "***"

    def get(self) -> str:
        return self._value


def _json_default(o: Any) -> Any:
    if isinstance(o, _Secret):
        return o.get()
    return str(o)


@dataclass
class RetrieverConfig:
    """Central configuration for every retrieval pipeline step."""

    # PostgreSQL (pgvector)
    database_url: str = ""

    # Embedding
    embedding_model_name: str = project_config.EMBEDDING_MODEL_NAME
    embed_cache_size: int = 256

    # Retrieval
    n_dense: int = project_config.RETRIEVAL_VECTOR_TOP_K
    n_sparse: int = project_config.RETRIEVAL_FTS_TOP_K
    rrf_k: int = project_config.RETRIEVAL_RRF_K
    top_k_after_fusion: int = 30
    final_top_k: int = project_config.RETRIEVAL_RERANK_TOP_N
    bm25_cache_dir: str = "bm25_cache"

    # Reranking
    reranker_model_name: str = project_config.CROSS_ENCODER_MODEL_NAME
    rerank_enabled: bool = True

    # Context expansion
    neighbor_chunks: int = 1
    max_context_tokens: int = 3000

    # LLM (Groq)
    llm_model: str = project_config.GROQ_MODEL
    llm_temperature: float = project_config.GROQ_TEMPERATURE
    llm_max_tokens: int = project_config.GROQ_MAX_TOKENS
    llm_context_window: int = project_config.GROQ_CONTEXT_WINDOW
    prompt_token_reserve: int = 64
    groq_api_key: _Secret = field(default_factory=lambda: _Secret(""))

    # Redis cache
    redis_host: str = "localhost"
    redis_port: int = 6379
    cache_ttl_seconds: int = 3600

    # Query preprocessing
    min_query_length: int = 3
    max_query_length: int = 2000

    # API
    api_key: _Secret = field(default_factory=lambda: _Secret(""))
    cors_allow_origins: list[str] = field(default_factory=lambda: ["*"])
    async_mode: bool = False


def load_config() -> RetrieverConfig:
    """
    Build RetrieverConfig from project defaults + RAG_* environment variables.

    Example: RAG_FINAL_TOP_K=3 overrides final_top_k.
    """
    cfg = RetrieverConfig(
        database_url=project_config.DATABASE_URL,
        groq_api_key=_Secret(project_config.GROQ_API_KEY),
        api_key=_Secret(project_config.API_KEY),
    )

    env_map = {
        "RAG_DATABASE_URL": ("database_url", str),
        "RAG_EMBEDDING_MODEL_NAME": ("embedding_model_name", str),
        "RAG_EMBED_CACHE_SIZE": ("embed_cache_size", int),
        "RAG_N_DENSE": ("n_dense", int),
        "RAG_N_SPARSE": ("n_sparse", int),
        "RAG_RRF_K": ("rrf_k", int),
        "RAG_TOP_K_AFTER_FUSION": ("top_k_after_fusion", int),
        "RAG_FINAL_TOP_K": ("final_top_k", int),
        "RAG_BM25_CACHE_DIR": ("bm25_cache_dir", str),
        "RAG_RERANKER_MODEL_NAME": ("reranker_model_name", str),
        "RAG_RERANK_ENABLED": ("rerank_enabled", lambda v: v.lower() in ("1", "true", "yes")),
        "RAG_NEIGHBOR_CHUNKS": ("neighbor_chunks", int),
        "RAG_MAX_CONTEXT_TOKENS": ("max_context_tokens", int),
        "RAG_LLM_MODEL": ("llm_model", str),
        "RAG_LLM_TEMPERATURE": ("llm_temperature", float),
        "RAG_LLM_MAX_TOKENS": ("llm_max_tokens", int),
        "RAG_LLM_CONTEXT_WINDOW": ("llm_context_window", int),
        "RAG_PROMPT_TOKEN_RESERVE": ("prompt_token_reserve", int),
        "RAG_GROQ_API_KEY": ("groq_api_key", str),
        "RAG_REDIS_HOST": ("redis_host", str),
        "RAG_REDIS_PORT": ("redis_port", int),
        "RAG_CACHE_TTL_SECONDS": ("cache_ttl_seconds", int),
        "RAG_MIN_QUERY_LENGTH": ("min_query_length", int),
        "RAG_MAX_QUERY_LENGTH": ("max_query_length", int),
    }

    for env_key, (attr, caster) in env_map.items():
        raw = os.getenv(env_key)
        if raw is not None and raw.strip() != "":
            value = caster(raw)
            if attr in ("groq_api_key", "api_key"):
                setattr(cfg, attr, _Secret(value))
            else:
                setattr(cfg, attr, value)

    if not cfg.groq_api_key.get():
        cfg.groq_api_key = _Secret(project_config.GROQ_API_KEY)

    return cfg


# =============================================================================
# SHARED DATA STRUCTURES
# =============================================================================


class QueryType(str, Enum):
    EMPTY = "empty"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    GREETING = "greeting"
    ACKNOWLEDGMENT = "acknowledgment"
    QUESTION = "question"


@dataclass
class ProcessedQuery:
    original_query: str
    cleaned_query: str
    query_type: QueryType
    is_valid: bool
    rejection_reason: Optional[str] = None
    language: Optional[str] = None
    char_count: int = 0
    processed_at: str = field(default_factory=_utc_iso)


@dataclass
class EmbeddingResult:
    query: str
    embedding: list[float]
    cache_hit: bool
    embed_time_ms: float
    processed_at: str = field(default_factory=_utc_iso)


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict
    dense_score: Optional[float]
    sparse_score: Optional[float]
    rrf_score: float
    retrieval_rank: int


@dataclass
class HybridRetrievalResult:
    query: str
    chunks: list[RetrievedChunk]
    total_dense_hits: int
    total_sparse_hits: int
    total_after_fusion: int
    dense_time_ms: float
    sparse_time_ms: float
    fusion_time_ms: float
    processed_at: str = field(default_factory=_utc_iso)


@dataclass
class FilterResult:
    chunks_before: int
    chunks_after: int
    filters_applied: dict
    dropped_chunk_ids: list[str]
    filter_time_ms: float
    processed_at: str = field(default_factory=_utc_iso)


@dataclass
class RankedChunk:
    chunk_id: str
    text: str
    metadata: dict
    rrf_score: float
    rerank_score: float
    final_rank: int


@dataclass
class RerankerResult:
    query: str
    chunks: list[RankedChunk]
    rerank_time_ms: float
    model_used: str
    processed_at: str = field(default_factory=_utc_iso)


@dataclass
class ExpandedChunk:
    chunk_id: str
    core_text: str
    expanded_text: str
    metadata: dict
    rerank_score: float
    final_rank: int
    neighbors_added: int


@dataclass
class ContextBuilderResult:
    chunks: list[ExpandedChunk]
    total_char_count: int
    estimated_tokens: int
    token_budget_used: float
    expand_time_ms: float
    processed_at: str = field(default_factory=_utc_iso)


@dataclass
class BuiltPrompt:
    system_prompt: str
    user_prompt: str
    context_chunks: list[ExpandedChunk]
    estimated_tokens: int
    processed_at: str = field(default_factory=_utc_iso)


@dataclass
class LLMResponse:
    raw_text: str
    model_used: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    llm_time_ms: float
    processed_at: str = field(default_factory=_utc_iso)


@dataclass
class Citation:
    source_file: str
    page_number: Optional[int]
    chunk_id: str


@dataclass
class ProcessedResponse:
    answer: str
    citations: list[Citation]
    has_answer: bool
    is_hallucination_risk: bool
    raw_answer: str
    processed_at: str = field(default_factory=_utc_iso)


@dataclass
class PipelineResult:
    """Full pipeline output including timings and retrieval diagnostics."""

    answer: str
    citations: list[Citation]
    has_answer: bool
    is_hallucination_risk: bool
    raw_answer: str
    query_type: QueryType
    chunks_used: list[RankedChunk]
    latency_breakdown: dict[str, float]
    llm_context_chunks: list[ExpandedChunk] = field(default_factory=list)
    cached: bool = False
    is_error: bool = False
    error_message: Optional[str] = None
    processed_at: str = field(default_factory=_utc_iso)


def format_llm_context_for_display(chunks: list[ExpandedChunk]) -> str:
    """Format expanded context blocks exactly as assembled for the Groq user prompt."""
    if not chunks:
        return "(no chunks sent to LLM)"

    lines = [
        "",
        "=" * 80,
        f"CHUNKS SENT TO GROQ LLM ({len(chunks)})",
        "=" * 80,
    ]
    for chunk in sorted(chunks, key=lambda c: c.final_rank):
        meta = chunk.metadata
        source = meta.get("source_file", "unknown")
        page = meta.get("page_number", "?")
        section = meta.get("section", "") or "N/A"
        lines.append(
            f"\n--- Rank {chunk.final_rank} | chunk_id={chunk.chunk_id} | "
            f"source={source} | page={page} | section={section} | "
            f"neighbors_added={chunk.neighbors_added} ---"
        )
        lines.append(chunk.expanded_text)
    lines.append("=" * 80)
    return "\n".join(lines)


@dataclass
class EvalCase:
    question: str
    expected_chunk_ids: list[str]
    expected_answer_keywords: list[str]


@dataclass
class EvalResult:
    case: EvalCase
    retrieved_ids: list[str]
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float = 0.0
    answer_coverage: float = 0.0
    faithful: Optional[bool] = None


@dataclass
class EvalReport:
    cases: list[EvalResult]
    avg_recall_at_k: float
    avg_precision_at_k: float
    avg_mrr: float
    avg_ndcg_at_k: float
    avg_answer_coverage: float
    total_cases: int
    eval_time_ms: float


# =============================================================================
# HELPERS — normalize PostgreSQL rows into pipeline metadata dicts.
# =============================================================================


def _parse_json_field(value: Any) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _row_to_chunk_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Map a PostgreSQL chunk row to the retrieval metadata schema."""
    extra = _parse_json_field(row.get("metadata"))
    page_no = row.get("page_no")
    return {
        "chunk_id": row["chunk_id"],
        "doc_id": row["doc_id"],
        "source_file": row.get("file_name") or extra.get("source_file", ""),
        "page_number": page_no if page_no is not None else extra.get("page_number"),
        "section": extra.get("section", "") or "",
        "chunk_index": row.get("chunk_index"),
        "char_count": row.get("char_count") or len(row.get("text") or ""),
        "created_at": row.get("created_at", ""),
    }


_CHITCHAT_RESPONSES = {
    QueryType.GREETING: (
        "Hello! I can answer questions about your uploaded documents. "
        "What would you like to know?"
    ),
    QueryType.ACKNOWLEDGMENT: (
        "You're welcome! Ask another question whenever you're ready."
    ),
    QueryType.EMPTY: "Please enter a question about your documents.",
    QueryType.TOO_SHORT: "Your question is too short. Please provide more detail.",
    QueryType.TOO_LONG: "Your question is too long. Please shorten it and try again.",
}


# =============================================================================
# STEP 1 — QueryPreprocessor
# =============================================================================


class QueryPreprocessor:
    """
    STEP 1: Clean, validate, and classify raw user queries.

  We short-circuit junk/chit-chat queries here so we never spend time on
  embedding, BM25 search, PostgreSQL vector queries, reranking, or LLM token
  cost on input that doesn't need any of that.
    """

    _GREETING_PATTERNS = {
        "hi", "hello", "hey", "good morning", "good afternoon",
        "good evening", "yo", "hiya",
    }
    _ACKNOWLEDGMENT_PATTERNS = {
        "thanks", "thank you", "ok", "okay", "got it", "cool",
        "great", "alright", "sounds good",
    }
    _PROMPT_INJECTION_PATTERNS = (
        r"\bsystem\s*:",
        r"ignore\s+previous\s+instructions",
        r"<\s*/?\s*inst\s*>",
    )

    def __init__(self, config: RetrieverConfig) -> None:
        self.config = config
        self.logger = _get_logger("query_preprocessor")

    def process(self, raw_query: Optional[str]) -> ProcessedQuery:
        """
        Clean and classify a query. Never raises on bad user input.

        Args:
            raw_query: User text; None is treated as empty.

        Returns:
            ProcessedQuery with is_valid=True only for real questions.
        """
        start = time.perf_counter()

        if raw_query is not None and not isinstance(raw_query, str):
            raise TypeError(f"Expected str query, got {type(raw_query)}")

        original = raw_query or ""
        cleaned = self._clean(original)

        if not cleaned:
            result = self._reject(original, cleaned, QueryType.EMPTY, "Query is empty.")
        else:
            query_type = self._classify(cleaned)
            if query_type in (QueryType.GREETING, QueryType.ACKNOWLEDGMENT):
                result = ProcessedQuery(
                    original_query=original,
                    cleaned_query=cleaned,
                    query_type=query_type,
                    is_valid=False,
                    rejection_reason=f"Query classified as '{query_type.value}'.",
                    language=self._detect_language(cleaned),
                    char_count=len(cleaned),
                )
            elif len(cleaned) < self.config.min_query_length:
                result = self._reject(
                    original, cleaned, QueryType.TOO_SHORT,
                    f"Query shorter than minimum {self.config.min_query_length} characters.",
                )
            elif len(cleaned) > self.config.max_query_length:
                result = self._reject(
                    original, cleaned, QueryType.TOO_LONG,
                    f"Query exceeds maximum {self.config.max_query_length} characters.",
                )
            else:
                result = ProcessedQuery(
                    original_query=original,
                    cleaned_query=cleaned,
                    query_type=QueryType.QUESTION,
                    is_valid=True,
                    language=self._detect_language(cleaned),
                    char_count=len(cleaned),
                )

        elapsed = _ms_since(start)
        self.logger.info(
            "Preprocess: input=1 → output=1 | valid=%s type=%s | %.2fms",
            result.is_valid, result.query_type.value, elapsed,
        )
        return result

    def _clean(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = "".join(ch for ch in text if ch.isprintable() or ch in ("\n", "\t"))
        text = self._sanitize_prompt_injection(text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _sanitize_prompt_injection(self, text: str) -> str:
        sanitized = text
        for pattern in self._PROMPT_INJECTION_PATTERNS:
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
        return sanitized

    def _classify(self, cleaned_query: str) -> QueryType:
        normalized = cleaned_query.lower().strip(" !.?")
        if normalized in self._GREETING_PATTERNS:
            return QueryType.GREETING
        if normalized in self._ACKNOWLEDGMENT_PATTERNS:
            return QueryType.ACKNOWLEDGMENT
        return QueryType.QUESTION

    def _detect_language(self, text: str) -> Optional[str]:
        if not _LANGDETECT_AVAILABLE:
            return None
        try:
            return langdetect_detect(text)
        except Exception:
            return None

    def _reject(
        self, original: str, cleaned: str, query_type: QueryType, reason: str
    ) -> ProcessedQuery:
        self.logger.info("Query rejected (%s): %s", query_type.value, reason)
        return ProcessedQuery(
            original_query=original,
            cleaned_query=cleaned,
            query_type=query_type,
            is_valid=False,
            rejection_reason=reason,
            char_count=len(cleaned),
        )


# =============================================================================
# STEP 2 — QueryEmbedder
# =============================================================================


class QueryEmbedder:
    """STEP 2: Embed cleaned queries with the same model used at ingestion."""

    def __init__(self, config: RetrieverConfig) -> None:
        self.config = config
        self.logger = _get_logger("embedder")
        self.logger.info(
            "Loading embedding model: %s — first load ~3s, then stays in memory.",
            config.embedding_model_name,
        )
        self._model = get_embedding_model()
        # LRU cache: embedding is a neural forward pass (~10–50ms on CPU).
        # Caching gives sub-millisecond hits for repeated queries.
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_lock = threading.Lock()

    def embed(self, processed_query: ProcessedQuery) -> EmbeddingResult:
        """
        Convert a valid ProcessedQuery into a dense embedding vector.

        Args:
            processed_query: Must have is_valid=True.

        Returns:
            EmbeddingResult with vector and timing.
        """
        start = time.perf_counter()
        assert processed_query.is_valid, "Cannot embed invalid query"

        query = processed_query.cleaned_query
        with self._cache_lock:
            if query in self._cache:
                self._cache.move_to_end(query)
                self.logger.debug("Embedding cache HIT for query hash prefix: %s", query[:40])
                return EmbeddingResult(
                    query=query,
                    embedding=self._cache[query],
                    cache_hit=True,
                    embed_time_ms=_ms_since(start),
                )

        self.logger.debug("Embedding cache MISS — computing vector.")
        vectors = self._model.encode([query])
        embedding = vectors[0].tolist() if hasattr(vectors[0], "tolist") else list(vectors[0])

        with self._cache_lock:
            self._cache[query] = embedding
            self._cache.move_to_end(query)
            while len(self._cache) > self.config.embed_cache_size:
                evicted_key, _ = self._cache.popitem(last=False)
                self.logger.debug("Embedding cache eviction: %s", evicted_key[:40])

        elapsed = _ms_since(start)
        self.logger.info("Embed: input=1 → output=1 | cache_hit=False | %.2fms", elapsed)
        return EmbeddingResult(
            query=query,
            embedding=embedding,
            cache_hit=False,
            embed_time_ms=elapsed,
        )


# =============================================================================
# STEP 3 — HybridRetriever (PostgreSQL pgvector + BM25)
# =============================================================================


class HybridRetriever:
    """
    STEP 3: Dense pgvector search + sparse BM25, fused with RRF.

    Dense leg uses PostgreSQL pgvector (cosine). Sparse leg uses BM25 over
    all chunk texts loaded from PostgreSQL at startup.
    """

    _CORPUS_SQL = """
        SELECT c.chunk_id, c.text, c.doc_id, c.chunk_index, c.page_no,
               c.source, c.metadata, c.char_count, c.created_at, d.file_name
        FROM chunks c
        JOIN documents d ON d.doc_id = c.doc_id
        ORDER BY c.doc_id, c.chunk_index
    """

    _CORPUS_SQL_PAGE = """
        SELECT c.chunk_id, c.text, c.doc_id, c.chunk_index, c.page_no,
               c.source, c.metadata, c.char_count, c.created_at, d.file_name
        FROM chunks c
        JOIN documents d ON d.doc_id = c.doc_id
        ORDER BY c.doc_id, c.chunk_index
        LIMIT %s OFFSET %s
    """

    _CORPUS_SQL_PAGE_ASYNC = """
        SELECT c.chunk_id, c.text, c.doc_id, c.chunk_index, c.page_no,
               c.source, c.metadata, c.char_count, c.created_at, d.file_name
        FROM chunks c
        JOIN documents d ON d.doc_id = c.doc_id
        ORDER BY c.doc_id, c.chunk_index
        LIMIT $1 OFFSET $2
    """

    def __init__(
        self,
        config: RetrieverConfig,
        store: PostgresStore | AsyncPostgresStore,
        corpus_rows: list[dict[str, Any]] | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self._async = isinstance(store, AsyncPostgresStore)
        self.logger = _get_logger("hybrid_retriever")

        if corpus_rows is None:
            if self._async:
                raise RuntimeError("Use HybridRetriever.create() for AsyncPostgresStore")
            load_start = time.perf_counter()
            cache_data, from_cache = self._load_corpus_paged(store, config)  # type: ignore[arg-type]
            self._init_corpus_from_cache(cache_data, from_cache, _ms_since(load_start))
        else:
            self._init_corpus(corpus_rows)

    @classmethod
    async def create(
        cls,
        config: RetrieverConfig,
        store: AsyncPostgresStore,
    ) -> HybridRetriever:
        logger = _get_logger("hybrid_retriever")
        logger.info("Loading BM25 corpus from PostgreSQL (async)...")
        load_start = time.perf_counter()
        cache_data, from_cache = await cls._aload_corpus_paged(store, config)
        self = cls.__new__(cls)
        self.config = config
        self.store = store
        self._async = True
        self.logger = logger
        self._init_corpus_from_cache(cache_data, from_cache, _ms_since(load_start))
        return self

    @staticmethod
    def _bm25_cache_path(config: RetrieverConfig) -> Path:
        db_hash = hashlib.md5(config.database_url.encode("utf-8")).hexdigest()
        cache_dir = Path(config.bm25_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{db_hash}.pkl"

    @classmethod
    def _read_bm25_disk_cache(cls, config: RetrieverConfig) -> dict[str, Any] | None:
        cache_path = cls._bm25_cache_path(config)
        if not cache_path.exists():
            return None
        if (time.time() - os.path.getmtime(cache_path)) >= 3600:
            return None
        with cache_path.open("rb") as handle:
            return pickle.load(handle)

    @classmethod
    def _write_bm25_disk_cache(cls, config: RetrieverConfig, cache_data: dict[str, Any]) -> None:
        cache_path = cls._bm25_cache_path(config)
        with cache_path.open("wb") as handle:
            pickle.dump(cache_data, handle)

    @classmethod
    def _rows_to_cache_data(cls, corpus_rows: list[dict[str, Any]]) -> dict[str, Any]:
        corpus_by_id = {row["chunk_id"]: row for row in corpus_rows}
        tokenized = [(row["text"] or "").lower().split() for row in corpus_rows]
        ids = [row["chunk_id"] for row in corpus_rows]
        return {"tokenized": tokenized, "ids": ids, "corpus_by_id": corpus_by_id}

    @classmethod
    def _load_corpus_paged(
        cls,
        store: PostgresStore,
        config: RetrieverConfig,
        page_size: int = 10_000,
    ) -> tuple[dict[str, Any], bool]:
        cached = cls._read_bm25_disk_cache(config)
        if cached is not None:
            return cached, True

        corpus_rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            batch = store.fetch_all(cls._CORPUS_SQL_PAGE, (page_size, offset))
            if not batch:
                break
            corpus_rows.extend(batch)
            if len(batch) < page_size:
                break
            offset += page_size

        cache_data = cls._rows_to_cache_data(corpus_rows)
        cls._write_bm25_disk_cache(config, cache_data)
        return cache_data, False

    @classmethod
    async def _aload_corpus_paged(
        cls,
        store: AsyncPostgresStore,
        config: RetrieverConfig,
        page_size: int = 10_000,
    ) -> tuple[dict[str, Any], bool]:
        cached = cls._read_bm25_disk_cache(config)
        if cached is not None:
            return cached, True

        corpus_rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            batch = await store.fetch_all(cls._CORPUS_SQL_PAGE_ASYNC, (page_size, offset))
            if not batch:
                break
            corpus_rows.extend(batch)
            if len(batch) < page_size:
                break
            offset += page_size

        cache_data = cls._rows_to_cache_data(corpus_rows)
        cls._write_bm25_disk_cache(config, cache_data)
        return cache_data, False

    def _init_corpus_from_cache(
        self,
        cache_data: dict[str, Any],
        from_cache: bool,
        load_ms: float,
    ) -> None:
        if not cache_data.get("ids"):
            raise RuntimeError(
                "PostgreSQL has no chunks yet. Run ingestion first, then retry retrieval. "
                "Check DATABASE_URL in your .env file."
            )

        self._corpus_by_id = cache_data["corpus_by_id"]
        self._bm25 = BM25Okapi(cache_data["tokenized"])
        self._bm25_ids = cache_data["ids"]
        self.logger.info(
            "BM25 corpus: %d chunks loaded in %.2fms (cache=%s)",
            len(self._bm25_ids),
            load_ms,
            from_cache,
        )

    def _init_corpus(self, corpus_rows: list[dict[str, Any]]) -> None:
        if not corpus_rows:
            raise RuntimeError(
                "PostgreSQL has no chunks yet. Run ingestion first, then retry retrieval. "
                "Check DATABASE_URL in your .env file."
            )

        cache_data = self._rows_to_cache_data(corpus_rows)
        self._init_corpus_from_cache(cache_data, from_cache=False, load_ms=0.0)

    def retrieve(self, query: str, query_embedding: list[float]) -> HybridRetrievalResult:
        """Run dense + sparse search and merge with RRF (sync)."""
        if self._async:
            raise RuntimeError("Use aretrieve() when backed by AsyncPostgresStore")
        dense_start = time.perf_counter()
        dense_chunks = self._dense_search(query_embedding)
        dense_ms = _ms_since(dense_start)

        sparse_start = time.perf_counter()
        sparse_chunks = self._sparse_search(query)
        sparse_ms = _ms_since(sparse_start)

        fusion_start = time.perf_counter()
        fused = self._rrf_merge(dense_chunks, sparse_chunks)
        fusion_ms = _ms_since(fusion_start)

        self.logger.info(
            "Retrieve: dense=%d sparse=%d → fused=%d | dense=%.2fms sparse=%.2fms fusion=%.2fms",
            len(dense_chunks), len(sparse_chunks), len(fused), dense_ms, sparse_ms, fusion_ms,
        )

        return HybridRetrievalResult(
            query=query,
            chunks=fused,
            total_dense_hits=len(dense_chunks),
            total_sparse_hits=len(sparse_chunks),
            total_after_fusion=len(fused),
            dense_time_ms=dense_ms,
            sparse_time_ms=sparse_ms,
            fusion_time_ms=fusion_ms,
        )

    async def aretrieve(
        self,
        query: str,
        query_embedding: list[float],
    ) -> HybridRetrievalResult:
        """Run dense + sparse search and merge with RRF (async)."""
        dense_start = time.perf_counter()
        dense_chunks = await self._adense_search(query_embedding)
        dense_ms = _ms_since(dense_start)

        sparse_start = time.perf_counter()
        sparse_chunks = await asyncio.to_thread(self._sparse_search, query)
        sparse_ms = _ms_since(sparse_start)

        fusion_start = time.perf_counter()
        fused = self._rrf_merge(dense_chunks, sparse_chunks)
        fusion_ms = _ms_since(fusion_start)

        self.logger.info(
            "Retrieve: dense=%d sparse=%d → fused=%d | dense=%.2fms sparse=%.2fms fusion=%.2fms",
            len(dense_chunks), len(sparse_chunks), len(fused), dense_ms, sparse_ms, fusion_ms,
        )

        return HybridRetrievalResult(
            query=query,
            chunks=fused,
            total_dense_hits=len(dense_chunks),
            total_sparse_hits=len(sparse_chunks),
            total_after_fusion=len(fused),
            dense_time_ms=dense_ms,
            sparse_time_ms=sparse_ms,
            fusion_time_ms=fusion_ms,
        )

    async def _adense_search(self, query_embedding: list[float]) -> list[RetrievedChunk]:
        rows = await self.store.vector_search(query_embedding, top_k=self.config.n_dense)  # type: ignore[union-attr]
        return self._rows_to_chunks(rows)

    def _dense_search(self, query_embedding: list[float]) -> list[RetrievedChunk]:
        """pgvector cosine search; score = similarity in [0, 1]."""
        rows = self.store.vector_search(query_embedding, top_k=self.config.n_dense)  # type: ignore[union-attr]
        return self._rows_to_chunks(rows)

    def _rows_to_chunks(self, rows: list[dict[str, Any]]) -> list[RetrievedChunk]:
        chunks: list[RetrievedChunk] = []
        for rank, row in enumerate(rows, start=1):
            dense_score = float(row.get("score") or 0.0)
            meta = _row_to_chunk_metadata(row)
            chunks.append(
                RetrievedChunk(
                    chunk_id=row["chunk_id"],
                    text=row["text"],
                    metadata=meta,
                    dense_score=dense_score,
                    sparse_score=None,
                    rrf_score=0.0,
                    retrieval_rank=rank,
                )
            )
        return chunks

    def _sparse_search(self, query_text: str) -> list[RetrievedChunk]:
        """
        BM25 (Best Match 25) keyword retrieval over the in-memory corpus.

        BM25 scores relevance from term frequency, inverse document frequency,
        and document length normalization. It has no semantic understanding —
        'car' and 'automobile' are unrelated — but excels at exact terms,
        codes, and rare keywords.
        """
        query_tokens = query_text.lower().split()
        all_scores = self._bm25.get_scores(query_tokens)
        if len(all_scores) == 0:
            return []

        max_score = float(np.max(all_scores))
        if max_score <= 0.0:
            return []

        top_indices = np.argsort(all_scores)[::-1][: self.config.n_sparse]
        chunks: list[RetrievedChunk] = []
        rank = 0
        for idx in top_indices:
            raw_score = float(all_scores[idx])
            if raw_score <= 0.0:
                break
            rank += 1
            chunk_id = self._bm25_ids[int(idx)]
            row = self._corpus_by_id[chunk_id]
            sparse_score = raw_score / max_score  # normalize to [0, 1]
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=row["text"],
                    metadata=_row_to_chunk_metadata(row),
                    dense_score=None,
                    sparse_score=sparse_score,
                    rrf_score=0.0,
                    retrieval_rank=rank,
                )
            )
        return chunks

    def _rrf_merge(
        self,
        dense_chunks: list[RetrievedChunk],
        sparse_chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """
        Reciprocal Rank Fusion (RRF) combines ranked lists:

            rrf_score = Σ 1 / (k + rank_i)

        k=60 dampens top-heavy lists. Chunks appearing in BOTH lists score
        higher than a chunk ranked #1 in only one list — consensus is signal.
        """
        k = self.config.rrf_k
        rrf_scores: dict[str, float] = {}
        dense_by_id: dict[str, RetrievedChunk] = {}
        sparse_by_id: dict[str, RetrievedChunk] = {}

        for rank, chunk in enumerate(dense_chunks, start=1):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            dense_by_id[chunk.chunk_id] = chunk

        for rank, chunk in enumerate(sparse_chunks, start=1):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            sparse_by_id[chunk.chunk_id] = chunk

        merged: list[RetrievedChunk] = []
        for chunk_id, score in sorted(rrf_scores.items(), key=lambda x: -x[1]):
            base = dense_by_id.get(chunk_id) or sparse_by_id[chunk_id]
            dense_score = dense_by_id[chunk_id].dense_score if chunk_id in dense_by_id else None
            sparse_score = sparse_by_id[chunk_id].sparse_score if chunk_id in sparse_by_id else None
            merged.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=base.text,
                    metadata=base.metadata,
                    dense_score=dense_score,
                    sparse_score=sparse_score,
                    rrf_score=score,
                    retrieval_rank=0,
                )
            )

        top = merged[: self.config.top_k_after_fusion]
        for i, chunk in enumerate(top, start=1):
            chunk.retrieval_rank = i
        return top


# =============================================================================
# STEP 4 — MetadataFilter
# =============================================================================


class MetadataFilter:
    """
    STEP 4: Post-fusion metadata filtering for scoping and access control.

    We filter after retrieval (not before) because BM25 ranks across all
    documents without native metadata filters. Filtering after fusion keeps
    both dense and sparse legs comparable under RRF.
    """

    def __init__(self) -> None:
        self.logger = _get_logger("metadata_filter")

    def apply(
        self,
        chunks: list[RetrievedChunk],
        filters: Optional[dict],
    ) -> tuple[list[RetrievedChunk], FilterResult]:
        start = time.perf_counter()
        before = len(chunks)

        if not filters:
            self.logger.info("No filters applied.")
            return chunks, FilterResult(
                chunks_before=before,
                chunks_after=before,
                filters_applied={},
                dropped_chunk_ids=[],
                filter_time_ms=_ms_since(start),
            )

        kept: list[RetrievedChunk] = []
        dropped_ids: list[str] = []

        for chunk in chunks:
            if self._passes(chunk.metadata, filters):
                kept.append(chunk)
            else:
                dropped_ids.append(chunk.chunk_id)
                self.logger.debug("Filter dropped chunk_id=%s", chunk.chunk_id)

        elapsed = _ms_since(start)
        if not kept:
            self.logger.warning("Filter removed ALL chunks. filters=%s", filters)

        self.logger.info(
            "Filter step: %d chunks in, %d chunks out (%d dropped) | %.2fms",
            before, len(kept), len(dropped_ids), elapsed,
        )
        return kept, FilterResult(
            chunks_before=before,
            chunks_after=len(kept),
            filters_applied=filters,
            dropped_chunk_ids=dropped_ids,
            filter_time_ms=elapsed,
        )

    def _passes(self, metadata: dict, filters: dict) -> bool:
        for key, expected in filters.items():
            if key not in metadata:
                return False
            actual = metadata[key]
            if isinstance(expected, list):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True


# =============================================================================
# STEP 5 — Reranker
# =============================================================================


class Reranker:
    """
    STEP 5: Cross-encoder reranking on a short candidate list.

    Bi-encoder (Step 2) encodes query and document separately — fast but
    less accurate. Cross-encoder jointly scores (query, document) pairs —
    slower but much more accurate. Industry pattern: retrieve-then-rerank.
    """

    def __init__(self, config: RetrieverConfig) -> None:
        self.config = config
        self.logger = _get_logger("reranker")
        load_start = time.perf_counter()
        self._model = CrossEncoder(config.reranker_model_name)
        # predict() is not safe for concurrent calls on one CrossEncoder instance.
        self._model_lock = threading.Lock()
        self.logger.info(
            "Cross-encoder loaded in %.2fms: %s",
            _ms_since(load_start), config.reranker_model_name,
        )

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> RerankerResult:
        start = time.perf_counter()
        if not chunks:
            return RerankerResult(
                query=query, chunks=[], rerank_time_ms=_ms_since(start),
                model_used=self.config.reranker_model_name,
            )

        try:
            pairs = [(query, c.text) for c in chunks]
            with self._model_lock:
                scores = self._model.predict(pairs)
            ranked = sorted(
                zip(chunks, scores),
                key=lambda x: float(x[1]),
                reverse=True,
            )[: self.config.final_top_k]

            output: list[RankedChunk] = []
            for rank, (chunk, score) in enumerate(ranked, start=1):
                output.append(
                    RankedChunk(
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                        metadata=chunk.metadata,
                        rrf_score=chunk.rrf_score,
                        rerank_score=float(score),
                        final_rank=rank,
                    )
                )
        except Exception:
            self.logger.exception("Reranker failed — falling back to RRF order.")
            fallback = sorted(chunks, key=lambda c: c.rrf_score, reverse=True)[
                : self.config.final_top_k
            ]
            output = [
                RankedChunk(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    metadata=c.metadata,
                    rrf_score=c.rrf_score,
                    rerank_score=c.rrf_score,
                    final_rank=i,
                )
                for i, c in enumerate(fallback, start=1)
            ]

        elapsed = _ms_since(start)
        self.logger.info(
            "Reranker: %d chunks in → %d chunks out | %.2fms",
            len(chunks), len(output), elapsed,
        )
        return RerankerResult(
            query=query,
            chunks=output,
            rerank_time_ms=elapsed,
            model_used=self.config.reranker_model_name,
        )


# =============================================================================
# STEP 6 — ContextBuilder
# =============================================================================


class ContextBuilder:
    """
    STEP 6: Expand top chunks with neighboring chunks from the same document.

    Semantic chunking can split mid-thought; neighbors restore missing context
    so the LLM sees complete passages. Token budget trimming drops lowest-ranked
    expansions first when context would exceed max_context_tokens.
    """

    def __init__(self, config: RetrieverConfig, store: PostgresStore | AsyncPostgresStore) -> None:
        self.config = config
        self.store = store
        self._async = isinstance(store, AsyncPostgresStore)
        self.logger = _get_logger("context_builder")

    def expand(self, ranked_chunks: list[RankedChunk]) -> ContextBuilderResult:
        if self._async:
            raise RuntimeError("Use aexpand() when backed by AsyncPostgresStore")
        return self._expand_impl(ranked_chunks)

    async def aexpand(self, ranked_chunks: list[RankedChunk]) -> ContextBuilderResult:
        return await self._expand_impl_async(ranked_chunks)

    def _expand_impl(self, ranked_chunks: list[RankedChunk]) -> ContextBuilderResult:
        start = time.perf_counter()
        expanded_list: list[ExpandedChunk] = []
        seen_text_keys: set[str] = set()

        for ranked in sorted(ranked_chunks, key=lambda c: c.final_rank):
            neighbors = self._fetch_neighbors(ranked.metadata)
            expanded_list.extend(
                self._build_expanded_entries(ranked, neighbors, seen_text_keys)
            )

        return self._finalize_expansion(ranked_chunks, expanded_list, start)

    async def _expand_impl_async(self, ranked_chunks: list[RankedChunk]) -> ContextBuilderResult:
        start = time.perf_counter()
        expanded_list: list[ExpandedChunk] = []
        seen_text_keys: set[str] = set()

        for ranked in sorted(ranked_chunks, key=lambda c: c.final_rank):
            neighbors = await self._afetch_neighbors(ranked.metadata)
            expanded_list.extend(
                self._build_expanded_entries(ranked, neighbors, seen_text_keys)
            )

        return self._finalize_expansion(ranked_chunks, expanded_list, start)

    def _build_expanded_entries(
        self,
        ranked: RankedChunk,
        neighbors: list[dict[str, Any]],
        seen_text_keys: set[str],
    ) -> list[ExpandedChunk]:
        parts = [n["text"] for n in neighbors]
        expanded_text = "\n".join(parts)
        text_key = expanded_text.strip()
        if text_key in seen_text_keys:
            self.logger.debug("Deduplicated overlapping expansion for %s", ranked.chunk_id)
            return []
        seen_text_keys.add(text_key)
        return [
            ExpandedChunk(
                chunk_id=ranked.chunk_id,
                core_text=ranked.text,
                expanded_text=expanded_text,
                metadata=ranked.metadata,
                rerank_score=ranked.rerank_score,
                final_rank=ranked.final_rank,
                neighbors_added=max(0, len(neighbors) - 1),
            )
        ]

    def _finalize_expansion(
        self,
        ranked_chunks: list[RankedChunk],
        expanded_list: list[ExpandedChunk],
        start: float,
    ) -> ContextBuilderResult:
        # Token budget: tiktoken-accurate count; drop lowest-priority chunks first.
        headroom = max_input_tokens(
            self.config.llm_context_window,
            self.config.llm_max_tokens,
            self.config.prompt_token_reserve,
        )
        budget = min(
            self.config.max_context_tokens,
            max(0, headroom - count_messages_tokens(_SYSTEM_PROMPT, "")),
        )
        before_trim = len(expanded_list)
        expanded_list, estimated_tokens = trim_chunks_to_budget(expanded_list, budget)
        trimmed = before_trim - len(expanded_list)
        total_chars = sum(len(c.expanded_text) for c in expanded_list)

        if trimmed:
            self.logger.warning(
                "Context exceeded token budget. Trimmed %d chunks.", trimmed
            )

        elapsed = _ms_since(start)
        self.logger.info(
            "Context expand: %d chunks → %d expanded | %d tokens (budget=%d) | %.2fms",
            len(ranked_chunks), len(expanded_list), estimated_tokens, budget, elapsed,
        )

        return ContextBuilderResult(
            chunks=sorted(expanded_list, key=lambda c: c.final_rank),
            total_char_count=total_chars,
            estimated_tokens=estimated_tokens,
            token_budget_used=(estimated_tokens / budget) if budget else 0.0,
            expand_time_ms=elapsed,
        )

    def _fetch_neighbors(self, metadata: dict) -> list[dict[str, Any]]:
        doc_id = metadata.get("doc_id")
        chunk_index = metadata.get("chunk_index")
        if doc_id is None or chunk_index is None:
            return self.store.fetch_all(  # type: ignore[union-attr]
                """
                SELECT c.text, c.chunk_id, c.chunk_index, c.page_no, c.metadata, d.file_name
                FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
                WHERE c.chunk_id = %s
                """,
                (metadata.get("chunk_id"),),
            )

        low = max(0, int(chunk_index) - self.config.neighbor_chunks)
        high = int(chunk_index) + self.config.neighbor_chunks
        return self.store.fetch_all(  # type: ignore[union-attr]
            """
            SELECT c.text, c.chunk_id, c.chunk_index, c.page_no, c.metadata, d.file_name
            FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
            WHERE c.doc_id = %s AND c.chunk_index BETWEEN %s AND %s
            ORDER BY c.chunk_index
            """,
            (doc_id, low, high),
        )

    async def _afetch_neighbors(self, metadata: dict) -> list[dict[str, Any]]:
        doc_id = metadata.get("doc_id")
        chunk_index = metadata.get("chunk_index")
        if doc_id is None or chunk_index is None:
            return await self.store.fetch_all(  # type: ignore[union-attr]
                """
                SELECT c.text, c.chunk_id, c.chunk_index, c.page_no, c.metadata, d.file_name
                FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
                WHERE c.chunk_id = $1
                """,
                (metadata.get("chunk_id"),),
            )

        low = max(0, int(chunk_index) - self.config.neighbor_chunks)
        high = int(chunk_index) + self.config.neighbor_chunks
        return await self.store.fetch_all(  # type: ignore[union-attr]
            """
            SELECT c.text, c.chunk_id, c.chunk_index, c.page_no, c.metadata, d.file_name
            FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
            WHERE c.doc_id = $1 AND c.chunk_index BETWEEN $2 AND $3
            ORDER BY c.chunk_index
            """,
            (doc_id, low, high),
        )


# =============================================================================
# STEP 7 — PromptBuilder
# =============================================================================


_SYSTEM_PROMPT = (
    "You are a precise, helpful assistant. Answer the user's question using "
    "ONLY the context provided below.\n"
    "Respond with valid JSON only (no markdown) using this schema:\n"
    "{\n"
    '  "answer": "string — your answer to the question",\n'
    '  "citations": [{"source": "filename.pdf", "page": 5}]\n'
    "}\n"
    "Rules:\n"
    "1. If the answer is in the context, put it in \"answer\" and cite sources in \"citations\".\n"
    "2. If the context partially answers, answer what you can and note gaps in \"answer\".\n"
    "3. If the context does not contain the answer, set answer to exactly: "
    "'I could not find an answer in the available documents.' and citations to [].\n"
    "4. Only cite sources that appear in the context. Do NOT guess or use outside knowledge.\n"
    "5. Keep the answer concise and factual."
)

_NO_ANSWER_PHRASE = "i could not find an answer in the available documents"


class PromptBuilder:
    """
    STEP 7: Assemble system + user prompts with cited context blocks.

    Strict context-only instructions reduce hallucination by forcing the LLM
    to ground answers in retrieved passages with verifiable citations.
    """

    def __init__(self, config: RetrieverConfig) -> None:
        self.config = config
        self.logger = _get_logger("prompt_builder")

    def build(
        self,
        processed_query: ProcessedQuery,
        context: ContextBuilderResult,
    ) -> BuiltPrompt:
        start = time.perf_counter()
        kept_chunks, user_prompt, estimated_tokens, was_clipped = fit_prompt_to_budget(
            system_prompt=_SYSTEM_PROMPT,
            question=processed_query.cleaned_query,
            chunks=context.chunks,
            context_window=self.config.llm_context_window,
            max_output_tokens=self.config.llm_max_tokens,
            reserve=self.config.prompt_token_reserve,
        )

        elapsed = _ms_since(start)
        if was_clipped:
            self.logger.warning(
                "Prompt clipped to fit model window: %d tokens (limit=%d).",
                estimated_tokens,
                max_input_tokens(
                    self.config.llm_context_window,
                    self.config.llm_max_tokens,
                    self.config.prompt_token_reserve,
                ),
            )
        self.logger.info(
            "Prompt build: %d context chunks | %d tokens | %.2fms",
            len(kept_chunks), estimated_tokens, elapsed,
        )

        return BuiltPrompt(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            context_chunks=kept_chunks,
            estimated_tokens=estimated_tokens,
        )


# =============================================================================
# STEP 8 — LLMClient
# =============================================================================


class LLMClient:
    """STEP 8: Groq chat completion via async httpx."""

    def __init__(self, config: RetrieverConfig) -> None:
        self.config = config
        self.logger = _get_logger("llm_client")
        api_key = config.groq_api_key.get()
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Add it to your .env file."
            )
        self._async_client = AsyncGroqClient(api_key=api_key)
        self._sync_client = SyncGroqClient(api_key=api_key)
        self._failure_count = 0
        self._open_until = 0.0
        self._breaker = (
            pybreaker.CircuitBreaker(fail_max=5, reset_timeout=60)
            if _PYBREAKER_AVAILABLE
            else None
        )

    async def aclose(self) -> None:
        await self._async_client.close()
        self._sync_client.close()

    def _prompt_messages(self, prompt: BuiltPrompt) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": prompt.user_prompt},
        ]

    def _check_prompt_budget(self, prompt: BuiltPrompt) -> None:
        limit = max_input_tokens(
            self.config.llm_context_window,
            self.config.llm_max_tokens,
            self.config.prompt_token_reserve,
        )
        prompt_tokens_est = count_messages_tokens(prompt.system_prompt, prompt.user_prompt)
        if prompt_tokens_est > limit:
            self.logger.warning(
                "Prompt still over budget after clipping (%d > %d); skipping LLM call.",
                prompt_tokens_est,
                limit,
            )
            raise RuntimeError(
                f"Prompt exceeds model context window ({prompt_tokens_est} > {limit} tokens)."
            )

    def _response_from_data(self, data: dict[str, Any], start: float) -> LLMResponse:
        choice = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", 0) or 0)
        elapsed = _ms_since(start)

        self.logger.info(
            "LLM: prompt_tokens=%d completion_tokens=%d | %.2fms",
            prompt_tokens, completion_tokens, elapsed,
        )

        return LLMResponse(
            raw_text=choice,
            model_used=self.config.llm_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            llm_time_ms=elapsed,
        )

    def generate(self, prompt: BuiltPrompt) -> LLMResponse:
        """Sync Groq call — safe inside Streamlit and other sync hosts."""
        start = time.perf_counter()
        if self._open_until and time.time() < self._open_until:
            raise RuntimeError("Groq circuit breaker open — skipping LLM call")

        self._check_prompt_budget(prompt)

        try:
            data = self._sync_client.chat_completion(
                model=self.config.llm_model,
                messages=self._prompt_messages(prompt),
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
                json_mode=True,
            )
            self._failure_count = 0
            self._open_until = 0.0
        except Exception as exc:
            self._record_groq_failure(exc)
            self.logger.exception("Groq API call failed.")
            if self._open_until and time.time() < self._open_until:
                raise RuntimeError(
                    "LLM unavailable — please try again shortly."
                ) from exc
            raise RuntimeError(f"LLM generation failed: {exc}") from exc

        return self._response_from_data(data, start)

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception(
            lambda exc: isinstance(exc, GroqAPIError) and exc.status_code in (429, 503)
        ),
        reraise=True,
    )
    async def _agenerate_with_retry(self, prompt: BuiltPrompt) -> dict[str, Any]:
        return await self._async_client.chat_completion(
            model=self.config.llm_model,
            messages=self._prompt_messages(prompt),
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
            json_mode=True,
        )

    def _record_groq_failure(self, exc: Exception) -> None:
        if isinstance(exc, GroqAPIError) and exc.status_code in (429, 503):
            self._failure_count += 1
            if self._failure_count >= 5:
                self._open_until = time.time() + 60

    async def agenerate(self, prompt: BuiltPrompt) -> LLMResponse:
        start = time.perf_counter()
        if self._open_until and time.time() < self._open_until:
            raise RuntimeError("Groq circuit breaker open — skipping LLM call")

        self._check_prompt_budget(prompt)

        try:
            data = await self._agenerate_with_retry(prompt)
            self._failure_count = 0
            self._open_until = 0.0
        except Exception as exc:
            self._record_groq_failure(exc)
            self.logger.exception("Groq API call failed.")
            if self._open_until and time.time() < self._open_until:
                raise RuntimeError(
                    "LLM unavailable — please try again shortly."
                ) from exc
            raise RuntimeError(f"LLM generation failed: {exc}") from exc

        return self._response_from_data(data, start)


# =============================================================================
# STEP 9 — ResponseProcessor
# =============================================================================


class ResponseProcessor:
    """STEP 9: Parse structured JSON LLM output and validate citations."""

    def __init__(self) -> None:
        self.logger = _get_logger("response_processor")

    @staticmethod
    def _increment_json_parse_failures() -> None:
        global _JSON_PARSE_FAILURES
        with _JSON_PARSE_LOCK:
            _JSON_PARSE_FAILURES += 1

    @staticmethod
    def json_parse_failure_count() -> int:
        with _JSON_PARSE_LOCK:
            return _JSON_PARSE_FAILURES

    def _parse_llm_json(self, raw: str) -> dict[str, Any] | None:
        def _validate(data: dict[str, Any]) -> bool:
            if not _JSONSCHEMA_AVAILABLE:
                return True
            try:
                jsonschema.validate(instance=data, schema=LLM_RESPONSE_SCHEMA)
                return True
            except jsonschema.ValidationError:
                return False

        try:
            data = json.loads(raw)
            if isinstance(data, dict) and _validate(data):
                return data
            return None
        except json.JSONDecodeError:
            repaired = raw
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                repaired = raw[start : end + 1]
            try:
                data = json.loads(repaired)
                if isinstance(data, dict) and _validate(data):
                    return data
            except json.JSONDecodeError:
                return None
        return None

    def process(
        self,
        llm_response: LLMResponse,
        context: ContextBuilderResult,
    ) -> ProcessedResponse:
        start = time.perf_counter()
        raw = (llm_response.raw_text or "").strip()

        answer = raw
        citations: list[Citation] = []
        hallucination_risk = False

        data = self._parse_llm_json(raw)
        if data is not None:
            answer = str(data.get("answer", raw)).strip()
            citations, hallucination_risk = self._parse_citations(
                data.get("citations", []),
                context,
            )
        else:
            self._increment_json_parse_failures()
            self.logger.warning(
                "LLM returned non-JSON; using raw text as answer. raw[:200]=%s",
                raw[:200],
            )
            answer = raw
            hallucination_risk = True

        has_answer = _NO_ANSWER_PHRASE not in answer.lower()

        elapsed = _ms_since(start)
        self.logger.info(
            "Postprocess: citations=%d hallucination_risk=%s | %.2fms",
            len(citations), hallucination_risk, elapsed,
        )

        return ProcessedResponse(
            answer=answer,
            citations=citations,
            has_answer=has_answer,
            is_hallucination_risk=hallucination_risk,
            raw_answer=raw,
        )

    def _parse_citations(
        self,
        raw_citations: Any,
        context: ContextBuilderResult,
    ) -> tuple[list[Citation], bool]:
        known_sources = {
            (c.metadata.get("source_file") or "").strip().lower()
            for c in context.chunks
        }
        known_chunk_ids = {c.chunk_id for c in context.chunks}
        citations: list[Citation] = []
        hallucination_risk = False

        if not isinstance(raw_citations, list):
            return citations, hallucination_risk

        for item in raw_citations:
            if not isinstance(item, dict):
                continue
            source_file = str(item.get("source", item.get("source_file", ""))).strip()
            page_raw = item.get("page", item.get("page_number"))
            page_number = int(page_raw) if page_raw is not None else None
            source_key = source_file.lower()

            matched_id = ""
            for chunk in context.chunks:
                if (chunk.metadata.get("source_file") or "").strip().lower() == source_key:
                    matched_id = chunk.chunk_id
                    break

            if source_key and source_key not in known_sources:
                hallucination_risk = True
                self.logger.warning(
                    "Citation source not in retrieved context: %s", source_file
                )

            citations.append(
                Citation(
                    source_file=source_file,
                    page_number=page_number,
                    chunk_id=matched_id or (next(iter(known_chunk_ids), "")),
                )
            )

        return citations, hallucination_risk


# =============================================================================
# CACHE — Redis query result cache
# =============================================================================


class QueryCache:
    """
    Redis cache for full pipeline results.

    We hash the CLEANED query (after preprocessing) so spacing/casing variants
    share one cache entry.
    """

    def __init__(self, config: RetrieverConfig) -> None:
        self.config = config
        self.logger = _get_logger("cache")
        self._client: Any = None
        self._async_client: Any = None

    async def connect(self) -> None:
        if not _REDIS_AVAILABLE:
            self.logger.warning("redis package not installed — caching disabled.")
            return
        try:
            import redis.asyncio as aioredis

            self._async_client = aioredis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            await self._async_client.ping()
            self.logger.info(
                "Redis async cache connected at %s:%d",
                self.config.redis_host,
                self.config.redis_port,
            )
        except Exception as exc:
            self.logger.warning("Redis unavailable — caching disabled: %s", exc)
            self._async_client = None

    async def aclose(self) -> None:
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    def _init_sync_client(self) -> None:
        if self._client is not None or not _REDIS_AVAILABLE:
            return
        try:
            self._client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            self._client.ping()
            self.logger.info(
                "Redis cache connected at %s:%d",
                self.config.redis_host,
                self.config.redis_port,
            )
        except Exception as exc:
            self.logger.warning("Redis unavailable — caching disabled: %s", exc)
            self._client = None

    @staticmethod
    def _cache_key(query: str, metadata_filters: Optional[dict] = None) -> str:
        canonical_filters = json.dumps(metadata_filters or {}, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(f"{query}|{canonical_filters}".encode("utf-8")).hexdigest()
        return f"rag:query:{digest}"

    def get(self, query: str, metadata_filters: Optional[dict] = None) -> Optional[PipelineResult]:
        self._init_sync_client()
        if not self._client:
            return None
        return self._deserialize(self._client.get(self._cache_key(query, metadata_filters)), query)

    async def aget(self, query: str, metadata_filters: Optional[dict] = None) -> Optional[PipelineResult]:
        if not self._async_client:
            return None
        raw = await self._async_client.get(self._cache_key(query, metadata_filters))
        return self._deserialize(raw, query)

    def _deserialize(self, raw: Any, query: str) -> Optional[PipelineResult]:
        if not raw:
            return None
        try:
            data = json.loads(raw)
            citations = [Citation(**c) for c in data.get("citations", [])]
            chunks_used = [RankedChunk(**c) for c in data.get("chunks_used", [])]
            llm_context_chunks = [
                ExpandedChunk(**c) for c in data.get("llm_context_chunks", [])
            ]
            self.logger.info("Cache HIT for query hash %s", self._cache_key(query)[-12:])
            return PipelineResult(
                answer=data["answer"],
                citations=citations,
                has_answer=data["has_answer"],
                is_hallucination_risk=data["is_hallucination_risk"],
                raw_answer=data["raw_answer"],
                query_type=QueryType(data["query_type"]),
                chunks_used=chunks_used,
                latency_breakdown=data.get("latency_breakdown", {}),
                llm_context_chunks=llm_context_chunks,
                cached=True,
                is_error=data.get("is_error", False),
                error_message=data.get("error_message"),
                processed_at=data.get("processed_at", _utc_iso()),
            )
        except Exception as exc:
            self.logger.warning("Cache read failed: %s", exc)
            return None

    def set(
        self,
        query: str,
        response: PipelineResult,
        metadata_filters: Optional[dict] = None,
    ) -> None:
        self._init_sync_client()
        if not self._client:
            return
        try:
            payload = asdict(response)
            payload["query_type"] = response.query_type.value
            self._client.setex(
                self._cache_key(query, metadata_filters),
                self.config.cache_ttl_seconds,
                json.dumps(payload, default=_json_default),
            )
        except Exception as exc:
            self.logger.warning("Cache write failed: %s", exc)

    async def aset(
        self,
        query: str,
        response: PipelineResult,
        metadata_filters: Optional[dict] = None,
    ) -> None:
        if not self._async_client:
            return
        try:
            payload = asdict(response)
            payload["query_type"] = response.query_type.value
            await self._async_client.setex(
                self._cache_key(query, metadata_filters),
                self.config.cache_ttl_seconds,
                json.dumps(payload, default=_json_default),
            )
        except Exception as exc:
            self.logger.warning("Cache write failed: %s", exc)

    def invalidate(self, query: str, metadata_filters: Optional[dict] = None) -> None:
        if not self._client:
            return
        try:
            self._client.delete(self._cache_key(query, metadata_filters))
        except Exception as exc:
            self.logger.warning("Cache invalidate failed: %s", exc)

    async def aflush_pattern(self, pattern: str = "rag:query:*") -> int:
        """Delete all keys matching pattern. Returns count deleted."""
        if not self._async_client:
            return 0
        keys = await self._async_client.keys(pattern)
        if not keys:
            return 0
        return int(await self._async_client.delete(*keys))


# =============================================================================
# PIPELINE ORCHESTRATOR
# =============================================================================


class RAGPipeline:
    """Wire Steps 1–9 into one callable retrieval + generation pipeline."""

    def __init__(self, config: Optional[RetrieverConfig] = None) -> None:
        self.config = config or load_config()
        self.logger = _get_logger("pipeline")

        if not self.config.database_url:
            raise RuntimeError(
                "DATABASE_URL is not configured. Set POSTGRES_* vars or DATABASE_URL in .env."
            )

        self.store = PostgresStore(self.config.database_url)
        try:
            self.preprocessor = QueryPreprocessor(self.config)
            self.embedder = QueryEmbedder(self.config)
            self.retriever = HybridRetriever(self.config, self.store)
            self.metadata_filter = MetadataFilter()
            self.reranker = Reranker(self.config)
            self.context_builder = ContextBuilder(self.config, self.store)
            self.prompt_builder = PromptBuilder(self.config)
            self.llm_client = LLMClient(self.config)
            self.response_processor = ResponseProcessor()
            self.cache = QueryCache(self.config)
            self.async_mode = False
        except Exception:
            self.store.close()
            raise

    @classmethod
    async def create(cls, config: Optional[RetrieverConfig] = None) -> RAGPipeline:
        """Async factory for API server — opens pools and loads corpus without blocking."""
        self = cls.__new__(cls)
        self.config = config or load_config()
        self.config.async_mode = True
        self.logger = _get_logger("pipeline")
        self.async_mode = True

        if not self.config.database_url:
            raise RuntimeError(
                "DATABASE_URL is not configured. Set POSTGRES_* vars or DATABASE_URL in .env."
            )

        self.store = AsyncPostgresStore(self.config.database_url)
        try:
            await self.store.open()
            self.preprocessor = QueryPreprocessor(self.config)
            self.embedder = QueryEmbedder(self.config)
            self.retriever = await HybridRetriever.create(self.config, self.store)
            self.metadata_filter = MetadataFilter()
            self.reranker = Reranker(self.config)
            self.context_builder = ContextBuilder(self.config, self.store)
            self.prompt_builder = PromptBuilder(self.config)
            self.llm_client = LLMClient(self.config)
            self.response_processor = ResponseProcessor()
            self.cache = QueryCache(self.config)
            await self.cache.connect()
        except Exception:
            if isinstance(self.store, AsyncPostgresStore):
                await self.store.close()
            raise

        self.logger.info("RAGPipeline ready (async mode).")
        return self

    def close(self) -> None:
        if isinstance(self.store, PostgresStore):
            self.store.close()
        try:
            if self.cache._client is not None:
                self.cache._client.close()
        except Exception:
            pass

    def __enter__(self) -> RAGPipeline:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    async def aclose(self) -> None:
        if isinstance(self.store, AsyncPostgresStore):
            await self.store.close()
        await self.cache.aclose()
        await self.llm_client.aclose()

    def run(
        self,
        raw_query: str,
        metadata_filters: Optional[dict] = None,
        rerank: bool = True,
        use_cache: bool = True,
    ) -> PipelineResult:
        """
        Execute the full pipeline and return a PipelineResult.

        Args:
            raw_query: User question text.
            metadata_filters: Optional metadata AND/OR filters.
            rerank: Whether to run cross-encoder reranking.
            use_cache: Whether to read/write Redis cache.

        Returns:
            PipelineResult with answer, citations, and latency breakdown.
        """
        request_id = _ensure_request_id()
        total_start = time.perf_counter()
        breakdown: dict[str, float] = {}

        try:
            # Step 1
            t0 = time.perf_counter()
            processed = self.preprocessor.process(raw_query)
            breakdown["preprocess_ms"] = _ms_since(t0)

            if use_cache and processed.is_valid:
                cached = self.cache.get(processed.cleaned_query, metadata_filters)
                if cached:
                    cached.latency_breakdown = {"total_ms": 0.0, "cache_hit": 1.0}
                    return cached

            if not processed.is_valid:
                answer = _CHITCHAT_RESPONSES.get(
                    processed.query_type,
                    processed.rejection_reason or "Invalid query.",
                )
                breakdown["total_ms"] = _ms_since(total_start)
                return PipelineResult(
                    answer=answer,
                    citations=[],
                    has_answer=False,
                    is_hallucination_risk=False,
                    raw_answer=answer,
                    query_type=processed.query_type,
                    chunks_used=[],
                    latency_breakdown=breakdown,
                )

            # Step 2
            t0 = time.perf_counter()
            embedding = self.embedder.embed(processed)
            breakdown["embed_ms"] = _ms_since(t0)

            # Step 3
            t0 = time.perf_counter()
            retrieval = self.retriever.retrieve(processed.cleaned_query, embedding.embedding)
            breakdown["retrieve_ms"] = (
                retrieval.dense_time_ms + retrieval.sparse_time_ms + retrieval.fusion_time_ms
            )

            # Step 4
            t0 = time.perf_counter()
            filtered_chunks, _ = self.metadata_filter.apply(retrieval.chunks, metadata_filters)
            breakdown["filter_ms"] = _ms_since(t0)

            if not filtered_chunks:
                answer = "I could not find an answer in the available documents."
                breakdown["total_ms"] = _ms_since(total_start)
                return PipelineResult(
                    answer=answer,
                    citations=[],
                    has_answer=False,
                    is_hallucination_risk=False,
                    raw_answer=answer,
                    query_type=processed.query_type,
                    chunks_used=[],
                    latency_breakdown=breakdown,
                )

            # Step 5
            t0 = time.perf_counter()
            if rerank and self.config.rerank_enabled:
                reranked = self.reranker.rerank(processed.cleaned_query, filtered_chunks)
                ranked_chunks = reranked.chunks
            else:
                ranked_chunks = [
                    RankedChunk(
                        chunk_id=c.chunk_id,
                        text=c.text,
                        metadata=c.metadata,
                        rrf_score=c.rrf_score,
                        rerank_score=c.rrf_score,
                        final_rank=i,
                    )
                    for i, c in enumerate(
                        sorted(filtered_chunks, key=lambda x: x.rrf_score, reverse=True)[
                            : self.config.final_top_k
                        ],
                        start=1,
                    )
                ]
            breakdown["rerank_ms"] = _ms_since(t0)

            # Step 6
            t0 = time.perf_counter()
            context = self.context_builder.expand(ranked_chunks)
            breakdown["expand_ms"] = _ms_since(t0)

            # Step 7
            t0 = time.perf_counter()
            prompt = self.prompt_builder.build(processed, context)
            breakdown["prompt_ms"] = _ms_since(t0)

            # Step 8
            t0 = time.perf_counter()
            try:
                llm_response = self.llm_client.generate(prompt)
            except RuntimeError as exc:
                if "circuit breaker" in str(exc).lower() or "LLM unavailable" in str(exc):
                    breakdown["llm_ms"] = _ms_since(t0)
                    breakdown["total_ms"] = _ms_since(total_start)
                    _observe_pipeline_metrics(breakdown)
                    return PipelineResult(
                        answer="LLM is temporarily unavailable. Please try again shortly.",
                        citations=[],
                        has_answer=False,
                        is_hallucination_risk=False,
                        raw_answer="",
                        query_type=processed.query_type,
                        chunks_used=ranked_chunks,
                        llm_context_chunks=prompt.context_chunks,
                        latency_breakdown=breakdown,
                        is_error=True,
                        error_message="LLM unavailable — please try again shortly.",
                    )
                raise
            breakdown["llm_ms"] = _ms_since(t0)

            # Step 9
            t0 = time.perf_counter()
            processed_response = self.response_processor.process(llm_response, context)
            breakdown["postprocess_ms"] = _ms_since(t0)

            breakdown["total_ms"] = _ms_since(total_start)

            result = PipelineResult(
                answer=processed_response.answer,
                citations=processed_response.citations,
                has_answer=processed_response.has_answer,
                is_hallucination_risk=processed_response.is_hallucination_risk,
                raw_answer=processed_response.raw_answer,
                query_type=processed.query_type,
                chunks_used=ranked_chunks,
                llm_context_chunks=prompt.context_chunks,
                latency_breakdown=breakdown,
            )

            if use_cache:
                self.cache.set(processed.cleaned_query, result, metadata_filters)

            _observe_pipeline_metrics(breakdown)
            return result

        except Exception as exc:
            self.logger.exception("Pipeline failed for query: %s", raw_query)
            breakdown["total_ms"] = _ms_since(total_start)
            return PipelineResult(
                answer="An error occurred while processing your question.",
                citations=[],
                has_answer=False,
                is_hallucination_risk=False,
                raw_answer="",
                query_type=QueryType.QUESTION,
                chunks_used=[],
                latency_breakdown=breakdown,
                is_error=True,
                error_message=str(exc),
            )

    async def arun(
        self,
        raw_query: str,
        metadata_filters: Optional[dict] = None,
        rerank: bool = True,
        use_cache: bool = True,
    ) -> PipelineResult:
        """Execute the full pipeline asynchronously (non-blocking I/O)."""
        if not self.async_mode:
            raise RuntimeError("Use RAGPipeline.create() for async execution")

        _ensure_request_id()
        total_start = time.perf_counter()
        breakdown: dict[str, float] = {}

        try:
            t0 = time.perf_counter()
            processed = self.preprocessor.process(raw_query)
            breakdown["preprocess_ms"] = _ms_since(t0)

            if use_cache and processed.is_valid:
                cached = await self.cache.aget(processed.cleaned_query, metadata_filters)
                if cached:
                    cached.latency_breakdown = {"total_ms": 0.0, "cache_hit": 1.0}
                    return cached

            if not processed.is_valid:
                answer = _CHITCHAT_RESPONSES.get(
                    processed.query_type,
                    processed.rejection_reason or "Invalid query.",
                )
                breakdown["total_ms"] = _ms_since(total_start)
                return PipelineResult(
                    answer=answer,
                    citations=[],
                    has_answer=False,
                    is_hallucination_risk=False,
                    raw_answer=answer,
                    query_type=processed.query_type,
                    chunks_used=[],
                    latency_breakdown=breakdown,
                )

            t0 = time.perf_counter()
            embedding = await asyncio.to_thread(self.embedder.embed, processed)
            breakdown["embed_ms"] = _ms_since(t0)

            t0 = time.perf_counter()
            retrieval = await self.retriever.aretrieve(
                processed.cleaned_query, embedding.embedding
            )
            breakdown["retrieve_ms"] = (
                retrieval.dense_time_ms + retrieval.sparse_time_ms + retrieval.fusion_time_ms
            )

            t0 = time.perf_counter()
            filtered_chunks, _ = self.metadata_filter.apply(retrieval.chunks, metadata_filters)
            breakdown["filter_ms"] = _ms_since(t0)

            if not filtered_chunks:
                answer = "I could not find an answer in the available documents."
                breakdown["total_ms"] = _ms_since(total_start)
                return PipelineResult(
                    answer=answer,
                    citations=[],
                    has_answer=False,
                    is_hallucination_risk=False,
                    raw_answer=answer,
                    query_type=processed.query_type,
                    chunks_used=[],
                    latency_breakdown=breakdown,
                )

            t0 = time.perf_counter()
            if rerank and self.config.rerank_enabled:
                reranked = await asyncio.to_thread(
                    self.reranker.rerank, processed.cleaned_query, filtered_chunks
                )
                ranked_chunks = reranked.chunks
            else:
                ranked_chunks = [
                    RankedChunk(
                        chunk_id=c.chunk_id,
                        text=c.text,
                        metadata=c.metadata,
                        rrf_score=c.rrf_score,
                        rerank_score=c.rrf_score,
                        final_rank=i,
                    )
                    for i, c in enumerate(
                        sorted(filtered_chunks, key=lambda x: x.rrf_score, reverse=True)[
                            : self.config.final_top_k
                        ],
                        start=1,
                    )
                ]
            breakdown["rerank_ms"] = _ms_since(t0)

            t0 = time.perf_counter()
            context = await self.context_builder.aexpand(ranked_chunks)
            breakdown["expand_ms"] = _ms_since(t0)

            t0 = time.perf_counter()
            prompt = self.prompt_builder.build(processed, context)
            breakdown["prompt_ms"] = _ms_since(t0)

            t0 = time.perf_counter()
            try:
                llm_response = await self.llm_client.agenerate(prompt)
            except RuntimeError as exc:
                if "circuit breaker" in str(exc).lower() or "LLM unavailable" in str(exc):
                    breakdown["llm_ms"] = _ms_since(t0)
                    breakdown["total_ms"] = _ms_since(total_start)
                    _observe_pipeline_metrics(breakdown)
                    return PipelineResult(
                        answer="LLM is temporarily unavailable. Please try again shortly.",
                        citations=[],
                        has_answer=False,
                        is_hallucination_risk=False,
                        raw_answer="",
                        query_type=processed.query_type,
                        chunks_used=ranked_chunks,
                        llm_context_chunks=prompt.context_chunks,
                        latency_breakdown=breakdown,
                        is_error=True,
                        error_message="LLM unavailable — please try again shortly.",
                    )
                raise
            breakdown["llm_ms"] = _ms_since(t0)

            t0 = time.perf_counter()
            processed_response = self.response_processor.process(llm_response, context)
            breakdown["postprocess_ms"] = _ms_since(t0)

            breakdown["total_ms"] = _ms_since(total_start)

            result = PipelineResult(
                answer=processed_response.answer,
                citations=processed_response.citations,
                has_answer=processed_response.has_answer,
                is_hallucination_risk=processed_response.is_hallucination_risk,
                raw_answer=processed_response.raw_answer,
                query_type=processed.query_type,
                chunks_used=ranked_chunks,
                llm_context_chunks=prompt.context_chunks,
                latency_breakdown=breakdown,
            )

            if use_cache:
                await self.cache.aset(processed.cleaned_query, result, metadata_filters)

            _observe_pipeline_metrics(breakdown)
            return result

        except Exception as exc:
            self.logger.exception("Pipeline failed for query: %s", raw_query)
            breakdown["total_ms"] = _ms_since(total_start)
            return PipelineResult(
                answer="An error occurred while processing your question.",
                citations=[],
                has_answer=False,
                is_hallucination_risk=False,
                raw_answer="",
                query_type=QueryType.QUESTION,
                chunks_used=[],
                latency_breakdown=breakdown,
                is_error=True,
                error_message=str(exc),
            )


# =============================================================================
# STEP 11 — PipelineEvaluator
# =============================================================================


class PipelineEvaluator:
    """
    Offline retrieval quality evaluation.

    Recall@K: fraction of relevant chunks found in top-K.
    Precision@K: fraction of top-K that are relevant.
    MRR: average 1/rank of the first correct hit (higher = good results on top).
    NDCG@K: discounted cumulative gain normalized against ideal ranking.
    answer_coverage: fraction of expected keywords present in the generated answer.
    """

    def __init__(self, pipeline: RAGPipeline) -> None:
        self.pipeline = pipeline
        self.logger = _get_logger("evaluator")

    @staticmethod
    def _ndcg(retrieved_ids: list[str], expected: set[str], k: int) -> float:
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, cid in enumerate(retrieved_ids[:k], start=1)
            if cid in expected
        )
        ideal_hits = min(len(expected), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        return dcg / idcg if idcg > 0 else 0.0

    async def _check_faithfulness(self, answer: str, context_snippet: str) -> bool:
        prompt = BuiltPrompt(
            system_prompt="Reply with only yes or no.",
            user_prompt=(
                f"Does this answer: '{answer}' only use information from these passages: "
                f"'{context_snippet}'? Reply yes or no."
            ),
            context_chunks=[],
            estimated_tokens=0,
        )
        try:
            response = await self.pipeline.llm_client.agenerate(prompt)
            reply = (response.raw_text or "").strip().lower()
            return reply.startswith("yes")
        except Exception:
            return False

    def evaluate(self, cases: list[EvalCase], k: int = 5) -> EvalReport:
        start = time.perf_counter()
        results: list[EvalResult] = []

        for case in cases:
            processed = self.pipeline.preprocessor.process(case.question)
            if not processed.is_valid:
                results.append(
                    EvalResult(case=case, retrieved_ids=[], recall_at_k=0.0,
                               precision_at_k=0.0, mrr=0.0, ndcg_at_k=0.0,
                               answer_coverage=0.0)
                )
                continue

            embedding = self.pipeline.embedder.embed(processed)
            retrieval = self.pipeline.retriever.retrieve(
                processed.cleaned_query, embedding.embedding
            )
            retrieved_ids = [c.chunk_id for c in retrieval.chunks[:k]]
            expected = set(case.expected_chunk_ids)
            retrieved_set = set(retrieved_ids)

            hits = expected & retrieved_set
            recall = len(hits) / len(expected) if expected else 0.0
            precision = len(hits) / len(retrieved_ids) if retrieved_ids else 0.0
            ndcg = self._ndcg(retrieved_ids, expected, k)

            mrr = 0.0
            for rank, chunk_id in enumerate(retrieved_ids, start=1):
                if chunk_id in expected:
                    mrr = 1.0 / rank
                    break

            # Optional end-to-end answer keyword check.
            pipeline_result = self.pipeline.run(case.question, use_cache=False)
            answer_lower = pipeline_result.answer.lower()
            if case.expected_answer_keywords:
                found = sum(1 for kw in case.expected_answer_keywords if kw.lower() in answer_lower)
                coverage = found / len(case.expected_answer_keywords)
            else:
                coverage = 1.0 if pipeline_result.has_answer else 0.0

            results.append(
                EvalResult(
                    case=case,
                    retrieved_ids=retrieved_ids,
                    recall_at_k=recall,
                    precision_at_k=precision,
                    mrr=mrr,
                    ndcg_at_k=ndcg,
                    answer_coverage=coverage,
                )
            )

        n = len(results) or 1
        report = EvalReport(
            cases=results,
            avg_recall_at_k=sum(r.recall_at_k for r in results) / n,
            avg_precision_at_k=sum(r.precision_at_k for r in results) / n,
            avg_mrr=sum(r.mrr for r in results) / n,
            avg_ndcg_at_k=sum(r.ndcg_at_k for r in results) / n,
            avg_answer_coverage=sum(r.answer_coverage for r in results) / n,
            total_cases=len(results),
            eval_time_ms=_ms_since(start),
        )
        self.logger.info(
            "Eval complete: cases=%d recall@k=%.3f precision@k=%.3f mrr=%.3f ndcg@k=%.3f | %.2fms",
            report.total_cases, report.avg_recall_at_k,
            report.avg_precision_at_k, report.avg_mrr, report.avg_ndcg_at_k, report.eval_time_ms,
        )
        return report

    async def async_evaluate(
        self,
        cases: list[EvalCase],
        k: int = 5,
        check_faithfulness: bool = False,
    ) -> EvalReport:
        start = time.perf_counter()
        results: list[EvalResult] = []

        for case in cases:
            processed = self.pipeline.preprocessor.process(case.question)
            if not processed.is_valid:
                results.append(
                    EvalResult(case=case, retrieved_ids=[], recall_at_k=0.0,
                               precision_at_k=0.0, mrr=0.0, ndcg_at_k=0.0,
                               answer_coverage=0.0)
                )
                continue

            embedding = await asyncio.to_thread(self.pipeline.embedder.embed, processed)
            retrieval = await self.pipeline.retriever.aretrieve(
                processed.cleaned_query, embedding.embedding
            )
            retrieved_ids = [c.chunk_id for c in retrieval.chunks[:k]]
            expected = set(case.expected_chunk_ids)
            retrieved_set = set(retrieved_ids)

            hits = expected & retrieved_set
            recall = len(hits) / len(expected) if expected else 0.0
            precision = len(hits) / len(retrieved_ids) if retrieved_ids else 0.0
            ndcg = self._ndcg(retrieved_ids, expected, k)

            mrr = 0.0
            for rank, chunk_id in enumerate(retrieved_ids, start=1):
                if chunk_id in expected:
                    mrr = 1.0 / rank
                    break

            pipeline_result = await self.pipeline.arun(case.question, use_cache=False)
            answer_lower = pipeline_result.answer.lower()
            if case.expected_answer_keywords:
                found = sum(1 for kw in case.expected_answer_keywords if kw.lower() in answer_lower)
                coverage = found / len(case.expected_answer_keywords)
            else:
                coverage = 1.0 if pipeline_result.has_answer else 0.0

            faithful: bool | None = None
            if check_faithfulness:
                context_snippet = " ".join(
                    c.expanded_text[:500] for c in pipeline_result.llm_context_chunks[:3]
                )
                faithful = await self._check_faithfulness(pipeline_result.answer, context_snippet)

            results.append(
                EvalResult(
                    case=case,
                    retrieved_ids=retrieved_ids,
                    recall_at_k=recall,
                    precision_at_k=precision,
                    mrr=mrr,
                    ndcg_at_k=ndcg,
                    answer_coverage=coverage,
                    faithful=faithful,
                )
            )

        n = len(results) or 1
        report = EvalReport(
            cases=results,
            avg_recall_at_k=sum(r.recall_at_k for r in results) / n,
            avg_precision_at_k=sum(r.precision_at_k for r in results) / n,
            avg_mrr=sum(r.mrr for r in results) / n,
            avg_ndcg_at_k=sum(r.ndcg_at_k for r in results) / n,
            avg_answer_coverage=sum(r.answer_coverage for r in results) / n,
            total_cases=len(results),
            eval_time_ms=_ms_since(start),
        )
        self.logger.info(
            "Async eval complete: cases=%d recall@k=%.3f precision@k=%.3f mrr=%.3f ndcg@k=%.3f | %.2fms",
            report.total_cases, report.avg_recall_at_k,
            report.avg_precision_at_k, report.avg_mrr, report.avg_ndcg_at_k, report.eval_time_ms,
        )
        return report


# =============================================================================
# STEP 10 — FastAPI production API (same file)
# =============================================================================

if _FASTAPI_AVAILABLE:

    from contextlib import asynccontextmanager

    ALLOWED_FILTER_KEYS = {"source_file", "doc_id", "page_number", "section"}

    class QueryRequest(BaseModel):
        query: str = Field(..., min_length=1, max_length=2000)
        metadata_filters: Optional[dict[str, Any]] = None
        rerank: bool = True

        @field_validator("metadata_filters")
        @classmethod
        def validate_filters(cls, v: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
            if v is None:
                return v
            unknown = set(v.keys()) - ALLOWED_FILTER_KEYS
            if unknown:
                raise ValueError(f"Unknown filter keys: {unknown}")
            return v

    class ChunkResponse(BaseModel):
        chunk_id: str
        source_file: str
        page_number: Optional[int]
        section: Optional[str]
        text: str
        final_rank: int
        rerank_score: Optional[float]

    class LLMContextChunkResponse(BaseModel):
        chunk_id: str
        source_file: str
        page_number: Optional[int]
        section: Optional[str]
        final_rank: int
        neighbors_added: int
        expanded_text: str

    class QueryResponse(BaseModel):
        answer: str
        citations: list[dict]
        has_answer: bool
        is_hallucination_risk: bool
        chunks_used: list[ChunkResponse]
        llm_context_chunks: list[LLMContextChunkResponse]
        latency_breakdown: dict
        cached: bool
        query_type: str

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        logger = _get_logger("api")
        logger.info("RAG Retrieval API starting up — loading models...")
        cfg = load_config()
        pipeline = await RAGPipeline.create(cfg)
        app.state.pipeline = pipeline
        app.state.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "latency_sums": {},
        }
        logger.info("RAG Retrieval API ready.")
        try:
            yield
        finally:
            await pipeline.aclose()

    app = FastAPI(title="RAG Retrieval API", version="1.0.0", lifespan=_lifespan)
    limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _result_to_response(result: PipelineResult) -> QueryResponse:
        chunks = [
            ChunkResponse(
                chunk_id=c.chunk_id,
                source_file=c.metadata.get("source_file", ""),
                page_number=c.metadata.get("page_number"),
                section=c.metadata.get("section") or None,
                text=c.text,
                final_rank=c.final_rank,
                rerank_score=c.rerank_score,
            )
            for c in result.chunks_used
        ]
        llm_chunks = [
            LLMContextChunkResponse(
                chunk_id=c.chunk_id,
                source_file=c.metadata.get("source_file", ""),
                page_number=c.metadata.get("page_number"),
                section=c.metadata.get("section") or None,
                final_rank=c.final_rank,
                neighbors_added=c.neighbors_added,
                expanded_text=c.expanded_text,
            )
            for c in result.llm_context_chunks
        ]
        return QueryResponse(
            answer=result.answer,
            citations=[asdict(c) for c in result.citations],
            has_answer=result.has_answer,
            is_hallucination_risk=result.is_hallucination_risk,
            chunks_used=chunks,
            llm_context_chunks=llm_chunks,
            latency_breakdown=result.latency_breakdown,
            cached=result.cached,
            query_type=result.query_type.value,
        )

    @app.post("/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
    @limiter.limit("10/minute")
    async def query_endpoint(request: Request, payload: QueryRequest) -> JSONResponse:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
        _REQUEST_ID_CTX.set(request_id[:8])

        pipeline: RAGPipeline = app.state.pipeline
        stats: dict = app.state.stats
        stats["total_queries"] += 1

        try:
            result = await pipeline.arun(
                raw_query=payload.query,
                metadata_filters=payload.metadata_filters,
                rerank=payload.rerank,
            )
            if result.cached:
                stats["cache_hits"] += 1
            for key, value in result.latency_breakdown.items():
                stats["latency_sums"][key] = stats["latency_sums"].get(key, 0.0) + value

            if result.is_error:
                raise HTTPException(
                    status_code=500,
                    detail={"error": result.error_message, "step": "pipeline"},
                )
            response = _result_to_response(result)
            return JSONResponse(
                content=response.model_dump(),
                headers={"X-Request-ID": _REQUEST_ID_CTX.get()},
            )
        except HTTPException:
            raise
        except Exception as exc:
            _get_logger("api").exception("POST /query failed")
            raise HTTPException(
                status_code=500,
                detail={"error": str(exc), "step": "api"},
            ) from exc

    @app.get("/health", dependencies=[Depends(verify_api_key)])
    @limiter.limit("30/minute")
    async def health(request: Request) -> dict:
        postgres_status = "disconnected"
        redis_status = "disconnected"
        pipeline: RAGPipeline | None = getattr(app.state, "pipeline", None)
        try:
            if pipeline is not None and isinstance(pipeline.store, AsyncPostgresStore):
                await pipeline.store.fetch_all("SELECT 1 AS ok")
                postgres_status = "connected"
        except Exception:
            pass
        try:
            if (
                pipeline is not None
                and _REDIS_AVAILABLE
                and pipeline.cache._async_client is not None
            ):
                await pipeline.cache._async_client.ping()
                redis_status = "connected"
        except Exception:
            pass
        bm25_status = (
            "loaded"
            if (
                pipeline is not None
                and hasattr(pipeline.retriever, "_bm25")
                and pipeline.retriever._bm25 is not None
            )
            else "not_loaded"
        )
        embedding_status = (
            "loaded"
            if (
                pipeline is not None
                and hasattr(pipeline.embedder, "_model")
                and pipeline.embedder._model is not None
            )
            else "not_loaded"
        )
        return {
            "status": "ok",
            "postgres": postgres_status,
            "redis": redis_status,
            "bm25_index": bm25_status,
            "embedding_model": embedding_status,
        }

    @app.get("/stats", dependencies=[Depends(verify_api_key)])
    @limiter.limit("30/minute")
    async def stats(request: Request) -> dict:
        s = getattr(app.state, "stats", {})
        total = s.get("total_queries", 0) or 1
        hits = s.get("cache_hits", 0)
        sums = s.get("latency_sums", {})
        avg_latency = {k: v / total for k, v in sums.items()}
        return {
            "total_queries": s.get("total_queries", 0),
            "cache_hit_rate": hits / total,
            "average_latency_per_step_ms": avg_latency,
            "json_parse_failure_count": ResponseProcessor.json_parse_failure_count(),
        }

    # Call after re-ingestion (HTTP POST) or subscribe to rag:ingest:complete pub/sub.
    @app.post("/cache/flush", dependencies=[Depends(verify_api_key)])
    @limiter.limit("5/minute")
    async def flush_cache(request: Request) -> dict:
        pipeline: RAGPipeline = app.state.pipeline
        count = await pipeline.cache.aflush_pattern()
        return {"flushed_keys": count}


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
    demo_logger = _get_logger("demo")

    cfg = load_config()
    pipeline = RAGPipeline(cfg)

    test_queries = [
        "What is the main topic of the uploaded documents?",
        "hi",
        "What is the Semantic chunking?",
    ]

    for q in test_queries:
        demo_logger.info("=" * 60)
        result = pipeline.run(q, use_cache=False)
        print(f"\nQuery                 : {q}")
        print(f"QueryType             : {result.query_type.value}")
        print(f"has_answer            : {result.has_answer}")
        print(f"answer (first 300)    : {result.answer[:300]}")
        print(f"citations             : {[asdict(c) for c in result.citations]}")
        print(f"is_hallucination_risk : {result.is_hallucination_risk}")
        print(f"latency_breakdown     : {result.latency_breakdown}")
        print(format_llm_context_for_display(result.llm_context_chunks))
