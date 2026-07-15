"""Shared pytest fixtures — set required env vars before project imports."""

from __future__ import annotations

import os

os.environ.setdefault("GROQ_API_KEY", "test-groq-key-for-pytest")
os.environ.setdefault("RAG_API_KEY", "test-api-key-for-pytest")
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql://pytest:pytest@localhost:5432/rag_db_test",
)
