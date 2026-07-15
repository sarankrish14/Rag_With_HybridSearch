"""
Public API for the RAG retrieval pipeline.

Import from here in application code:

    from retrieval import RAGPipeline, RetrieverConfig, load_config, create_rag_pipeline
"""

from __future__ import annotations

from typing import Optional

from retrieval.pipeline import (
    ExpandedChunk,
    PipelineResult,
    RAGPipeline,
    RetrieverConfig,
    format_llm_context_for_display,
    load_config,
)


def create_rag_pipeline(config: Optional[RetrieverConfig] = None) -> RAGPipeline:
    """Build a configured RAGPipeline instance."""
    return RAGPipeline(config or load_config())


__all__ = [
    "RAGPipeline",
    "RetrieverConfig",
    "load_config",
    "create_rag_pipeline",
    "PipelineResult",
    "ExpandedChunk",
    "format_llm_context_for_display",
]
