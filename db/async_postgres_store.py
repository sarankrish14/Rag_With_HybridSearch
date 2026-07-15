"""Async PostgreSQL access for retrieval (asyncpg + pgvector)."""

from __future__ import annotations

import json
import logging
from typing import Any

import asyncpg

logger = logging.getLogger(__name__)

try:
    from pgvector.asyncpg import register_vector

    _PGVECTOR_AVAILABLE = True
except ImportError:
    _PGVECTOR_AVAILABLE = False


class AsyncPostgresStore:
    """Async read-focused PostgreSQL store with connection pooling."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self._pool: asyncpg.Pool | None = None

    async def open(self) -> None:
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(
            self.database_url,
            min_size=1,
            max_size=10,
            command_timeout=30,
            init=self._init_connection,
        )
        logger.info("AsyncPostgresStore pool ready")

    @staticmethod
    async def _init_connection(conn: asyncpg.Connection) -> None:
        if _PGVECTOR_AVAILABLE:
            await register_vector(conn)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def fetch_all(
        self,
        sql: str,
        params: tuple[Any, ...] | list[Any] | None = None,
    ) -> list[dict[str, Any]]:
        if self._pool is None:
            raise RuntimeError("AsyncPostgresStore is not open")
        args = list(params or ())
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)
        return [dict(row) for row in rows]

    async def vector_search(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        doc_id: str | None = None,
    ) -> list[dict[str, Any]]:
        vec = [float(x) for x in query_embedding]
        doc_filter = "AND c.doc_id = $4" if doc_id else ""
        if doc_id:
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
                    1 - (c.embedding <=> $1::vector) AS score
                FROM chunks c
                JOIN documents d ON d.doc_id = c.doc_id
                WHERE TRUE {doc_filter}
                ORDER BY c.embedding <=> $2::vector
                LIMIT $3
            """
            params: tuple[Any, ...] = (vec, vec, top_k, doc_id)
        else:
            sql = """
                SELECT
                    c.chunk_id,
                    c.doc_id,
                    c.chunk_index,
                    c.text,
                    c.page_no,
                    c.source,
                    c.metadata,
                    d.file_name,
                    1 - (c.embedding <=> $1::vector) AS score
                FROM chunks c
                JOIN documents d ON d.doc_id = c.doc_id
                ORDER BY c.embedding <=> $2::vector
                LIMIT $3
            """
            params = (vec, vec, top_k)

        rows = await self.fetch_all(sql, params)
        for row in rows:
            meta = row.get("metadata")
            if isinstance(meta, str):
                try:
                    row["metadata"] = json.loads(meta)
                except json.JSONDecodeError:
                    row["metadata"] = {}
        return rows
