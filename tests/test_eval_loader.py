"""Tests for eval case loading."""

from __future__ import annotations

import json

import pytest

from evals.loader import load_eval_cases_from_data
from retrieval.pipeline import RetrieverConfig


@pytest.fixture
def config() -> RetrieverConfig:
    return RetrieverConfig(database_url="postgresql://pytest@localhost/rag_db_test")


def test_load_cases_with_explicit_chunk_ids(config: RetrieverConfig) -> None:
    data = {
        "cases": [
            {
                "question": "What is RAG?",
                "expected_chunk_ids": ["chunk-a", "chunk-b"],
                "expected_answer_keywords": ["retrieval", "generation"],
            }
        ]
    }
    cases = load_eval_cases_from_data(data, config)
    assert len(cases) == 1
    assert cases[0].question == "What is RAG?"
    assert cases[0].expected_chunk_ids == ["chunk-a", "chunk-b"]
    assert cases[0].expected_answer_keywords == ["retrieval", "generation"]


def test_load_cases_requires_question(config: RetrieverConfig) -> None:
    with pytest.raises(ValueError, match="question"):
        load_eval_cases_from_data({"cases": [{"expected_chunk_ids": ["x"]}]}, config)


def test_load_cases_empty_list_raises(config: RetrieverConfig) -> None:
    with pytest.raises(ValueError, match="No eval cases"):
        load_eval_cases_from_data({"cases": []}, config)
