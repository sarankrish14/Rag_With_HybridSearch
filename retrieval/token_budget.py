"""Accurate token counting and hard-clipping for LLM prompts (tiktoken)."""

from __future__ import annotations

import logging
from dataclasses import is_dataclass, replace
from functools import lru_cache
from typing import Any, Protocol

logger = logging.getLogger(__name__)

_ENCODING_NAME = "cl100k_base"
_CHAT_OVERHEAD_TOKENS = 8


class _TextChunk(Protocol):
    chunk_id: str
    expanded_text: str
    metadata: dict[str, Any]
    final_rank: int


@lru_cache(maxsize=1)
def _encoding():
    import tiktoken

    return tiktoken.get_encoding(_ENCODING_NAME)


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_encoding().encode(text))


def count_messages_tokens(system_prompt: str, user_prompt: str) -> int:
    return _CHAT_OVERHEAD_TOKENS + count_tokens(system_prompt) + count_tokens(user_prompt)


def max_input_tokens(
    context_window: int,
    max_output_tokens: int,
    reserve: int = 64,
) -> int:
    return max(0, context_window - max_output_tokens - reserve)


def clip_text_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0 or not text:
        return ""
    tokens = _encoding().encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _encoding().decode(tokens[:max_tokens])


def count_chunk_list_tokens(chunks: list[_TextChunk]) -> int:
    if not chunks:
        return 0
    parts = [c.expanded_text for c in sorted(chunks, key=lambda c: c.final_rank)]
    return count_tokens("\n\n".join(parts))


def trim_chunks_to_budget(
    chunks: list[_TextChunk],
    budget: int,
) -> tuple[list[_TextChunk], int]:
    """Drop lowest-priority chunks (highest final_rank) until text fits budget."""
    kept = sorted(chunks, key=lambda c: c.final_rank)
    while kept and count_chunk_list_tokens(kept) > budget:
        dropped = kept.pop()
        logger.debug("Token budget dropped chunk_id=%s (rank=%d)", dropped.chunk_id, dropped.final_rank)

    tokens = count_chunk_list_tokens(kept)
    return kept, tokens


def _format_context_block(chunk: _TextChunk) -> str:
    meta = chunk.metadata
    source = meta.get("source_file", "unknown")
    page = meta.get("page_number", "?")
    section = meta.get("section", "") or "N/A"
    return (
        f"--- Source: {source} | Page: {page} | Section: {section} ---\n"
        f"{chunk.expanded_text}\n-------"
    )


def build_user_prompt(question: str, blocks: list[str]) -> str:
    if blocks:
        body = "\n\n".join(blocks)
        return f"CONTEXT:\n{body}\n\nQuestion: {question}"
    return f"CONTEXT:\n\nQuestion: {question}"


def fit_prompt_to_budget(
    system_prompt: str,
    question: str,
    chunks: list[_TextChunk],
    context_window: int,
    max_output_tokens: int,
    reserve: int = 64,
) -> tuple[list[_TextChunk], str, int, bool]:
    """
    Fit context chunks + question into the model input window.

    Returns:
        (kept_chunks, user_prompt, total_prompt_tokens, was_clipped)
    """
    limit = max_input_tokens(context_window, max_output_tokens, reserve)
    ordered = sorted(chunks, key=lambda c: c.final_rank)
    kept: list[_TextChunk] = list(ordered)
    was_clipped = False

    while kept:
        blocks = [_format_context_block(c) for c in kept]
        user_prompt = build_user_prompt(question, blocks)
        total = count_messages_tokens(system_prompt, user_prompt)
        if total <= limit:
            return kept, user_prompt, total, was_clipped
        kept.pop()
        was_clipped = True

    if ordered:
        chunk = ordered[0]
        meta = chunk.metadata
        source = meta.get("source_file", "unknown")
        page = meta.get("page_number", "?")
        section = meta.get("section", "") or "N/A"
        header = f"--- Source: {source} | Page: {page} | Section: {section} ---\n"
        footer = "\n-------"
        overhead = count_messages_tokens(
            system_prompt,
            build_user_prompt(question, [f"{header}{footer}"]),
        )
        body_budget = max(0, limit - overhead)
        clipped_body = clip_text_to_tokens(chunk.expanded_text, body_budget)
        if clipped_body != chunk.expanded_text:
            was_clipped = True
            if is_dataclass(chunk):
                chunk = replace(chunk, expanded_text=clipped_body)
            else:
                chunk.expanded_text = clipped_body  # type: ignore[misc]
            kept = [chunk]

    blocks = [_format_context_block(c) for c in kept]
    user_prompt = build_user_prompt(question, blocks)
    total = count_messages_tokens(system_prompt, user_prompt)

    if total > limit:
        user_prompt = clip_text_to_tokens(user_prompt, max(0, limit - count_tokens(system_prompt) - _CHAT_OVERHEAD_TOKENS))
        total = count_messages_tokens(system_prompt, user_prompt)
        was_clipped = True
        logger.warning(
            "Hard-clipped full user prompt to %d tokens (limit=%d).",
            total,
            limit,
        )

    return kept, user_prompt, total, was_clipped
