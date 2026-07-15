"""API key authentication for FastAPI endpoints."""

from __future__ import annotations

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from settings import settings

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Security(API_KEY_HEADER)) -> str:
    if not api_key or api_key != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key
