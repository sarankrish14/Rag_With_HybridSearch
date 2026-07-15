"""Tests for the retrieval pipeline — mocked orchestration and component coverage."""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from retrieval.pipeline import (
    ContextBuilder,
    ContextBuilderResult,
    EmbeddingResult,
    EvalCase,
    ExpandedChunk,
    HybridRetrievalResult,
    HybridRetriever,
    LLMResponse,
    PipelineEvaluator,
    PipelineResult,
    ProcessedQuery,
    PromptBuilder,
    QueryCache,
    QueryPreprocessor,
    QueryType,
    RAGPipeline,
    RankedChunk,
    Reranker,
    ResponseProcessor,
    RetrievedChunk,
    RetrieverConfig,
    _CHITCHAT_RESPONSES,
    format_llm_context_for_display,
    load_config,
)


@pytest.fixture
def config() -> RetrieverConfig:
    return RetrieverConfig(
        database_url="postgresql://pytest@localhost/rag_db_test",
        groq_api_key="test-groq-key",
        min_query_length=3,
        max_query_length=200,
        rrf_k=60,
        top_k_after_fusion=10,
        final_top_k=3,
        neighbor_chunks=1,
        max_context_tokens=3000,
        rerank_enabled=True,
    )


@pytest.fixture
def preprocessor(config: RetrieverConfig) -> QueryPreprocessor:
    return QueryPreprocessor(config)


def _retrieved_chunk(
    chunk_id: str,
    text: str,
    *,
    source_file: str = "sample.pdf",
    rrf_score: float = 0.5,
    rank: int = 1,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        metadata={
            "source_file": source_file,
            "page_number": 1,
            "doc_id": "doc-1",
            "chunk_index": 0,
            "chunk_id": chunk_id,
        },
        dense_score=0.8,
        sparse_score=0.6,
        rrf_score=rrf_score,
        retrieval_rank=rank,
    )


def _ranked_chunk(chunk_id: str, text: str, *, rank: int = 1) -> RankedChunk:
    return RankedChunk(
        chunk_id=chunk_id,
        text=text,
        metadata={
            "source_file": "sample.pdf",
            "page_number": 3,
            "doc_id": "doc-1",
            "chunk_index": 1,
            "chunk_id": chunk_id,
        },
        rrf_score=0.5,
        rerank_score=0.9,
        final_rank=rank,
    )


def _expanded_context() -> ContextBuilderResult:
    return ContextBuilderResult(
        chunks=[
            ExpandedChunk(
                chunk_id="chunk-1",
                core_text="Warranty is 2 years.",
                expanded_text="Warranty is 2 years.",
                metadata={"source_file": "sample.pdf", "page_number": 3},
                rerank_score=0.9,
                final_rank=1,
                neighbors_added=0,
            )
        ],
        total_char_count=20,
        estimated_tokens=5,
        token_budget_used=0.1,
        expand_time_ms=1.0,
    )


def _hybrid_result(query: str, chunks: list[RetrievedChunk]) -> HybridRetrievalResult:
    return HybridRetrievalResult(
        query=query,
        chunks=chunks,
        total_dense_hits=len(chunks),
        total_sparse_hits=len(chunks),
        total_after_fusion=len(chunks),
        dense_time_ms=1.0,
        sparse_time_ms=1.0,
        fusion_time_ms=0.5,
    )


def _make_mock_pipeline(config: RetrieverConfig) -> RAGPipeline:
    """Build a RAGPipeline with mocked heavy dependencies (no DB/models/Groq)."""
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.config = config
    pipeline.logger = logging.getLogger("test.pipeline")
    pipeline.preprocessor = QueryPreprocessor(config)
    pipeline.embedder = MagicMock()
    pipeline.retriever = MagicMock()
    pipeline.metadata_filter = MagicMock()
    pipeline.reranker = MagicMock()
    pipeline.context_builder = MagicMock()
    pipeline.prompt_builder = PromptBuilder(config)
    pipeline.llm_client = MagicMock()
    pipeline.response_processor = ResponseProcessor()
    pipeline.cache = QueryCache(config)
    pipeline.async_mode = False
    pipeline.store = MagicMock()
    return pipeline


class TestQueryPreprocessorExtended:
    def test_none_query_treated_as_empty(self, preprocessor: QueryPreprocessor) -> None:
        result = preprocessor.process(None)
        assert result.is_valid is False
        assert result.query_type == QueryType.EMPTY

    def test_acknowledgment_blocked(self, preprocessor: QueryPreprocessor) -> None:
        result = preprocessor.process("thank you")
        assert result.is_valid is False
        assert result.query_type == QueryType.ACKNOWLEDGMENT

    def test_too_long_query_rejected(self, preprocessor: QueryPreprocessor) -> None:
        result = preprocessor.process("x" * 201)
        assert result.is_valid is False
        assert result.query_type == QueryType.TOO_LONG

    def test_whitespace_normalized(self, preprocessor: QueryPreprocessor) -> None:
        result = preprocessor.process("  What   is   RAG?  ")
        assert result.is_valid is True
        assert result.cleaned_query == "What is RAG?"

    def test_non_string_query_raises(self, preprocessor: QueryPreprocessor) -> None:
        with pytest.raises(TypeError):
            preprocessor.process(123)  # type: ignore[arg-type]


class TestMetadataFilterExtended:
    def test_page_number_filter(self) -> None:
        from retrieval.pipeline import MetadataFilter

        chunks = [
            _retrieved_chunk("1", "page one", rank=1),
            RetrievedChunk(
                chunk_id="2",
                text="page two",
                metadata={"source_file": "sample.pdf", "page_number": 2},
                dense_score=0.4,
                sparse_score=None,
                rrf_score=0.3,
                retrieval_rank=2,
            ),
        ]
        chunks[0].metadata["page_number"] = 1

        filt = MetadataFilter()
        kept, result = filt.apply(chunks, {"page_number": 2})

        assert [c.chunk_id for c in kept] == ["2"]
        assert result.chunks_before == 2
        assert result.chunks_after == 1

    def test_list_filter_matches_any_value(self) -> None:
        from retrieval.pipeline import MetadataFilter

        chunks = [
            RetrievedChunk(
                chunk_id="1",
                text="a",
                metadata={"source_file": "a.pdf"},
                dense_score=0.5,
                sparse_score=None,
                rrf_score=0.1,
                retrieval_rank=1,
            ),
            RetrievedChunk(
                chunk_id="2",
                text="b",
                metadata={"source_file": "b.pdf"},
                dense_score=0.4,
                sparse_score=None,
                rrf_score=0.05,
                retrieval_rank=2,
            ),
        ]
        filt = MetadataFilter()
        kept, _ = filt.apply(chunks, {"source_file": ["b.pdf", "c.pdf"]})

        assert [c.chunk_id for c in kept] == ["2"]

    def test_missing_metadata_key_drops_chunk(self) -> None:
        from retrieval.pipeline import MetadataFilter

        chunks = [
            RetrievedChunk(
                chunk_id="1",
                text="no doc id",
                metadata={"source_file": "sample.pdf"},
                dense_score=0.5,
                sparse_score=None,
                rrf_score=0.1,
                retrieval_rank=1,
            ),
        ]
        filt = MetadataFilter()
        kept, result = filt.apply(chunks, {"doc_id": "doc-1"})

        assert kept == []
        assert result.dropped_chunk_ids == ["1"]


class TestHybridRetrieverSparse:
    def _corpus_row(self, chunk_id: str, text: str) -> dict:
        return {
            "chunk_id": chunk_id,
            "text": text,
            "doc_id": "doc-1",
            "chunk_index": 0,
            "page_no": 1,
            "source": "sample.pdf",
            "metadata": "{}",
            "char_count": len(text),
            "created_at": "2024-01-01",
            "file_name": "sample.pdf",
        }

    def test_sparse_search_ranks_keyword_matches(self, config: RetrieverConfig) -> None:
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.config = config
        retriever.logger = logging.getLogger("test.retriever")
        retriever._async = False
        retriever._init_corpus(
            [
                self._corpus_row(
                    "c1",
                    "semantic chunking is a technique that splits documents by meaning "
                    "using semantic chunking boundaries",
                ),
                self._corpus_row(
                    "c2",
                    "warranty coverage lasts two years for products under warranty terms",
                ),
                self._corpus_row(
                    "c3",
                    "semantic analysis helps semantic chunking improve retrieval quality",
                ),
            ]
        )

        hits = retriever._sparse_search("semantic chunking")

        assert len(hits) >= 1
        assert hits[0].chunk_id in {"c1", "c3"}
        assert hits[0].sparse_score == pytest.approx(1.0)


class TestReranker:
    @patch("retrieval.pipeline.CrossEncoder")
    def test_reranker_orders_by_cross_encoder_score(
        self, mock_ce_cls: MagicMock, config: RetrieverConfig
    ) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.95]
        mock_ce_cls.return_value = mock_model

        reranker = Reranker(config)
        chunks = [
            _retrieved_chunk("low", "low relevance text", rrf_score=0.9, rank=1),
            _retrieved_chunk("high", "high relevance text", rrf_score=0.5, rank=2),
        ]

        result = reranker.rerank("what is the warranty?", chunks)

        assert result.chunks[0].chunk_id == "high"
        assert result.chunks[0].rerank_score == pytest.approx(0.95)
        assert result.chunks[0].final_rank == 1

    @patch("retrieval.pipeline.CrossEncoder")
    def test_reranker_empty_input(self, mock_ce_cls: MagicMock, config: RetrieverConfig) -> None:
        mock_ce_cls.return_value = MagicMock()
        reranker = Reranker(config)

        result = reranker.rerank("query", [])

        assert result.chunks == []


class TestContextBuilder:
    def test_expand_joins_neighbor_text(self, config: RetrieverConfig) -> None:
        store = MagicMock()
        store.fetch_all.return_value = [
            {"text": "Intro paragraph."},
            {"text": "Core answer paragraph."},
            {"text": "Follow-up paragraph."},
        ]
        builder = ContextBuilder(config, store)
        ranked = [_ranked_chunk("chunk-1", "Core answer paragraph.")]

        result = builder.expand(ranked)

        assert len(result.chunks) == 1
        expanded = result.chunks[0]
        assert "Intro paragraph." in expanded.expanded_text
        assert "Core answer paragraph." in expanded.expanded_text
        assert expanded.neighbors_added == 2

    def test_expand_deduplicates_overlapping_windows(self, config: RetrieverConfig) -> None:
        store = MagicMock()
        shared_neighbors = [{"text": "Same neighbor window."}]
        store.fetch_all.return_value = shared_neighbors
        builder = ContextBuilder(config, store)
        ranked = [
            _ranked_chunk("chunk-1", "first", rank=1),
            _ranked_chunk("chunk-2", "second", rank=2),
        ]

        result = builder.expand(ranked)

        assert len(result.chunks) == 1


class TestPromptBuilder:
    def test_build_includes_question_and_context(self, config: RetrieverConfig) -> None:
        builder = PromptBuilder(config)
        processed = ProcessedQuery(
            original_query="What is the warranty?",
            cleaned_query="What is the warranty?",
            query_type=QueryType.QUESTION,
            is_valid=True,
            char_count=22,
        )
        context = _expanded_context()

        prompt = builder.build(processed, context)

        assert "What is the warranty?" in prompt.user_prompt
        assert "Warranty is 2 years." in prompt.user_prompt
        assert prompt.estimated_tokens > 0
        assert len(prompt.context_chunks) == 1


class TestResponseProcessorExtended:
    def test_json_wrapped_in_markdown_is_parsed(self) -> None:
        processor = ResponseProcessor()
        payload = {
            "answer": "The warranty is 2 years.",
            "citations": [{"source": "sample.pdf", "page": 3}],
        }
        llm = LLMResponse(
            raw_text=f"```json\n{json.dumps(payload)}\n```",
            model_used="test",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            llm_time_ms=1.0,
        )

        result = processor.process(llm, _expanded_context())

        assert result.has_answer is True
        assert result.is_hallucination_risk is False
        assert result.answer == "The warranty is 2 years."

    def test_no_answer_phrase_sets_has_answer_false(self) -> None:
        processor = ResponseProcessor()
        llm = LLMResponse(
            raw_text=json.dumps(
                {
                    "answer": "I could not find an answer in the available documents.",
                    "citations": [],
                }
            ),
            model_used="test",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            llm_time_ms=1.0,
        )

        result = processor.process(llm, _expanded_context())

        assert result.has_answer is False
        assert result.is_hallucination_risk is False


class TestQueryCache:
    def test_cache_key_stable_for_same_query_and_filters(self, config: RetrieverConfig) -> None:
        key_a = QueryCache._cache_key("what is rag?", {"source_file": "a.pdf"})
        key_b = QueryCache._cache_key("what is rag?", {"source_file": "a.pdf"})
        key_c = QueryCache._cache_key("what is rag?", {"source_file": "b.pdf"})

        assert key_a == key_b
        assert key_a != key_c
        assert key_a.startswith("rag:query:")

    def test_deserialize_round_trip(self, config: RetrieverConfig) -> None:
        cache = QueryCache(config)
        original = PipelineResult(
            answer="Cached answer",
            citations=[],
            has_answer=True,
            is_hallucination_risk=False,
            raw_answer='{"answer":"Cached answer","citations":[]}',
            query_type=QueryType.QUESTION,
            chunks_used=[_ranked_chunk("c1", "text")],
            latency_breakdown={"total_ms": 12.0},
        )
        payload = json.dumps(
            {
                "answer": original.answer,
                "citations": [],
                "has_answer": True,
                "is_hallucination_risk": False,
                "raw_answer": original.raw_answer,
                "query_type": original.query_type.value,
                "chunks_used": [
                    {
                        "chunk_id": "c1",
                        "text": "text",
                        "metadata": _ranked_chunk("c1", "text").metadata,
                        "rrf_score": 0.5,
                        "rerank_score": 0.9,
                        "final_rank": 1,
                    }
                ],
                "latency_breakdown": {"total_ms": 12.0},
                "llm_context_chunks": [],
            }
        )

        restored = cache._deserialize(payload, "what is rag?")

        assert restored is not None
        assert restored.cached is True
        assert restored.answer == "Cached answer"
        assert restored.query_type == QueryType.QUESTION


class TestRAGPipelineRun:
    def test_greeting_short_circuits_without_retrieval(self, config: RetrieverConfig) -> None:
        pipeline = _make_mock_pipeline(config)

        result = pipeline.run("hello", use_cache=False)

        assert result.has_answer is False
        assert result.answer == _CHITCHAT_RESPONSES[QueryType.GREETING]
        pipeline.embedder.embed.assert_not_called()
        pipeline.retriever.retrieve.assert_not_called()

    def test_cache_hit_returns_without_retrieval(self, config: RetrieverConfig) -> None:
        pipeline = _make_mock_pipeline(config)
        cached = PipelineResult(
            answer="From cache",
            citations=[],
            has_answer=True,
            is_hallucination_risk=False,
            raw_answer="From cache",
            query_type=QueryType.QUESTION,
            chunks_used=[],
            latency_breakdown={},
            cached=True,
        )
        pipeline.cache.get = MagicMock(return_value=cached)

        result = pipeline.run("What is semantic chunking?", use_cache=True)

        assert result.answer == "From cache"
        assert result.latency_breakdown.get("cache_hit") == 1.0
        pipeline.retriever.retrieve.assert_not_called()

    def test_no_chunks_after_filter_returns_fallback(self, config: RetrieverConfig) -> None:
        pipeline = _make_mock_pipeline(config)
        pipeline.embedder.embed.return_value = EmbeddingResult(
            query="What is missing?",
            embedding=[0.1, 0.2],
            cache_hit=False,
            embed_time_ms=1.0,
        )
        pipeline.retriever.retrieve.return_value = _hybrid_result(
            "What is missing?",
            [_retrieved_chunk("c1", "text")],
        )
        pipeline.metadata_filter.apply.return_value = ([], MagicMock())

        result = pipeline.run("What is missing?", use_cache=False)

        assert result.has_answer is False
        assert "could not find an answer" in result.answer.lower()
        pipeline.reranker.rerank.assert_not_called()

    def test_full_happy_path(self, config: RetrieverConfig) -> None:
        pipeline = _make_mock_pipeline(config)
        pipeline.cache.get = MagicMock(return_value=None)
        pipeline.embedder.embed.return_value = EmbeddingResult(
            query="What is the warranty?",
            embedding=[0.1, 0.2, 0.3],
            cache_hit=False,
            embed_time_ms=2.0,
        )
        pipeline.retriever.retrieve.return_value = _hybrid_result(
            "What is the warranty?",
            [_retrieved_chunk("chunk-1", "Warranty is 2 years.")],
        )
        pipeline.metadata_filter.apply.return_value = (
            [_retrieved_chunk("chunk-1", "Warranty is 2 years.")],
            MagicMock(),
        )
        pipeline.reranker.rerank.return_value = MagicMock(
            chunks=[_ranked_chunk("chunk-1", "Warranty is 2 years.")]
        )
        context = _expanded_context()
        pipeline.context_builder.expand.return_value = context
        pipeline.llm_client.generate.return_value = LLMResponse(
            raw_text=json.dumps(
                {
                    "answer": "The warranty is 2 years.",
                    "citations": [{"source": "sample.pdf", "page": 3}],
                }
            ),
            model_used="test-model",
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            llm_time_ms=50.0,
        )

        result = pipeline.run("What is the warranty?", use_cache=False)

        assert result.has_answer is True
        assert result.is_hallucination_risk is False
        assert result.answer == "The warranty is 2 years."
        assert len(result.citations) == 1
        assert result.latency_breakdown.get("total_ms", 0) > 0
        pipeline.llm_client.generate.assert_called_once()


class TestPipelineEvaluator:
    def test_evaluate_computes_retrieval_metrics(self, config: RetrieverConfig) -> None:
        pipeline = _make_mock_pipeline(config)
        pipeline.embedder.embed.return_value = EmbeddingResult(
            query="warranty question",
            embedding=[0.5],
            cache_hit=False,
            embed_time_ms=1.0,
        )
        pipeline.retriever.retrieve.return_value = _hybrid_result(
            "warranty question",
            [
                _retrieved_chunk("expected-1", "warranty text", rank=1),
                _retrieved_chunk("other", "other text", rank=2),
            ],
        )
        pipeline.run = MagicMock(
            return_value=PipelineResult(
                answer="The warranty is 2 years.",
                citations=[],
                has_answer=True,
                is_hallucination_risk=False,
                raw_answer="",
                query_type=QueryType.QUESTION,
                chunks_used=[],
                latency_breakdown={},
            )
        )

        evaluator = PipelineEvaluator(pipeline)
        report = evaluator.evaluate(
            [
                EvalCase(
                    question="What is the warranty?",
                    expected_chunk_ids=["expected-1"],
                    expected_answer_keywords=["warranty", "years"],
                )
            ],
            k=2,
        )

        assert report.total_cases == 1
        assert report.avg_recall_at_k == pytest.approx(1.0)
        assert report.avg_precision_at_k == pytest.approx(0.5)
        assert report.avg_mrr == pytest.approx(1.0)
        assert report.avg_answer_coverage == pytest.approx(1.0)


class TestUtilities:
    def test_format_llm_context_for_display(self) -> None:
        chunks = [
            ExpandedChunk(
                chunk_id="c1",
                core_text="core",
                expanded_text="Expanded passage text.",
                metadata={
                    "source_file": "manual.pdf",
                    "page_number": 4,
                    "section": "Warranty",
                },
                rerank_score=0.8,
                final_rank=1,
                neighbors_added=1,
            )
        ]

        rendered = format_llm_context_for_display(chunks)

        assert "CHUNKS SENT TO GROQ LLM" in rendered
        assert "manual.pdf" in rendered
        assert "Expanded passage text." in rendered

    def test_format_llm_context_empty(self) -> None:
        assert "no chunks sent" in format_llm_context_for_display([]).lower()

    def test_load_config_reads_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAG_FINAL_TOP_K", "7")
        cfg = load_config()
        assert cfg.final_top_k == 7
