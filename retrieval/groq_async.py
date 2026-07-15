"""Async Groq chat completions via httpx."""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqAPIError(RuntimeError):
    """Groq API error with status code for retry decisions."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _chat_payload(
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    json_mode: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    return payload


def _raise_for_response(response: httpx.Response) -> dict[str, Any]:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.error("Groq API error %s: %s", exc.response.status_code, exc.response.text)
        raise GroqAPIError(
            f"Groq API failed: {exc.response.text}",
            status_code=exc.response.status_code,
        ) from exc
    return response.json()


class SyncGroqClient:
    """Synchronous Groq client for sync callers (CLI, Streamlit) — avoids asyncio loop issues."""

    def __init__(self, api_key: str, timeout: float = 60.0) -> None:
        self._client = httpx.Client(
            base_url="https://api.groq.com/openai/v1",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    def close(self) -> None:
        self._client.close()

    def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool = True,
    ) -> dict[str, Any]:
        payload = _chat_payload(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
        response = self._client.post("/chat/completions", json=payload)
        return _raise_for_response(response)


class AsyncGroqClient:
    def __init__(self, api_key: str, timeout: float = 60.0) -> None:
        self._api_key = api_key
        self._client = httpx.AsyncClient(
            base_url="https://api.groq.com/openai/v1",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool = True,
    ) -> dict[str, Any]:
        payload = _chat_payload(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
        response = await self._client.post("/chat/completions", json=payload)
        return _raise_for_response(response)
