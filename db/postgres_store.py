"""
PostgreSQL store for documents, chunk metadata, and embeddings.

Uses pgvector for 768-dimensional cosine similarity search.
Connection string is read from config.DATABASE_URL (or DATABASE_URL env var).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from pgvector.psycopg import register_vector

import config

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PostgresStore:
    """Persist documents and chunk embeddings in PostgreSQL."""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or config.DATABASE_URL
        if not self.database_url:
            raise ValueError(
                "DATABASE_URL is not set. Add it to .env, e.g. "
                "postgresql://user:password@localhost:5432/rag_db"
            )

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self.database_url, connect_timeout=5)
        register_vector(conn)
        return conn

    def init_db(self) -> None:
        """Create pgvector extension and tables from schema_postgres.sql."""
        schema_path = Path(__file__).parent / "schema_postgres.sql"
        schema_sql = schema_path.read_text(encoding="utf-8")
        with self._connect() as conn:
            conn.execute(schema_sql)
            conn.commit()
        logger.info("PostgreSQL schema initialized")

    def get_document_by_hash(self, file_hash: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM documents WHERE file_hash = %s",
                (file_hash,),
            )
            row = cur.fetchone()
            if not row:
                return None
            columns = [desc.name for desc in cur.description]
            return dict(zip(columns, row))

    def insert_document(
        self,
        doc_id: str,
        file_name: str,
        file_hash: str,
        file_path: str | None = None,
        page_count: int | None = None,
        docling_metadata: dict[str, Any] | None = None,
        status: str = "complete",
    ) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    doc_id, file_name, file_path, file_hash, page_count,
                    status, docling_metadata, created_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    doc_id,
                    file_name,
                    file_path,
                    file_hash,
                    page_count,
                    status,
                    json.dumps(docling_metadata) if docling_metadata else None,
                    now,
                    now,
                ),
            )
            conn.commit()

    def insert_chunks(self, doc_id: str, chunks: list[dict[str, Any]]) -> int:
        """Insert chunk rows with embeddings. Returns number of rows inserted."""
        if not chunks:
            return 0

        now = _utc_now()
        rows = []
        for chunk in chunks:
            extra_metadata = {
                k: v
                for k, v in chunk.items()
                if k not in {
                    "chunk_id", "doc_id", "chunk_index", "text", "embedding",
                    "sentence_count", "char_count", "page_no", "source", "threshold_used",
                }
            }
            rows.append(
                (
                    chunk["chunk_id"],
                    doc_id,
                    chunk["chunk_index"],
                    chunk["text"],
                    chunk["embedding"],
                    chunk.get("sentence_count"),
                    chunk.get("char_count"),
                    chunk.get("page_no"),
                    chunk.get("source"),
                    chunk.get("threshold_used"),
                    json.dumps(extra_metadata) if extra_metadata else None,
                    now,
                )
            )

        sql = """
            INSERT INTO chunks (
                chunk_id, doc_id, chunk_index, text, embedding,
                sentence_count, char_count, page_no, source,
                threshold_used, metadata, created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, rows)
            conn.commit()

        logger.info("Stored %d chunks for doc_id=%s", len(rows), doc_id)
        return len(rows)

    def delete_document(self, doc_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
            conn.commit()
        logger.info("Deleted document doc_id=%s", doc_id)

    def delete_document_by_hash(self, file_hash: str) -> str | None:
        """Delete document (and cascaded chunks) by file hash. Returns removed doc_id."""
        existing = self.get_document_by_hash(file_hash)
        if not existing:
            return None
        doc_id = existing["doc_id"]
        self.delete_document(doc_id)
        return doc_id

    @staticmethod
    def _rows_to_dicts(cur: psycopg.Cursor) -> list[dict[str, Any]]:
        columns = [desc.name for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]

    def fetch_all(
        self,
        sql: str,
        params: tuple[Any, ...] | list[Any] | None = None,
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            cur = conn.execute(sql, params or ())
            return self._rows_to_dicts(cur)

    def vector_search(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        doc_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Dense cosine-similarity search via pgvector HNSW index."""
        doc_filter = "AND c.doc_id = %s" if doc_id else ""
        params: list[Any] = [query_embedding, query_embedding]
        if doc_id:
            params.append(doc_id)
        params.append(top_k)

        sql = f"""
            SELECT
                c.chunk_id,
                c.doc_id,
                c.chunk_index,
                c.text,
                c.page_no,
                c.source,
                c.metadata,
                d.file_name,
                1 - (c.embedding <=> %s::vector) AS score
            FROM chunks c
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE TRUE {doc_filter}
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
        """
        return self.fetch_all(sql, params)

    def close(self) -> None:
        """Lifecycle hook — sync store opens a connection per request (no pool)."""
        pass

    def fts_search(
        self,
        query: str,
        top_k: int = 20,
        doc_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Sparse keyword search via PostgreSQL full-text search (tsvector)."""
        doc_filter = "AND c.doc_id = %s" if doc_id else ""
        params: list[Any] = [query]
        if doc_id:
            params.append(doc_id)
        params.append(top_k)

        sql = f"""
            SELECT
                c.chunk_id,
                c.doc_id,
                c.chunk_index,
                c.text,
                c.page_no,
                c.source,
                c.metadata,
                d.file_name,
                ts_rank_cd(c.text_tsv, q.query) AS score
            FROM chunks c
            JOIN documents d ON d.doc_id = c.doc_id
            CROSS JOIN plainto_tsquery('english', %s) AS q(query)
            WHERE c.text_tsv @@ q.query {doc_filter}
            ORDER BY score DESC
            LIMIT %s
        """
        return self.fetch_all(sql, params)
