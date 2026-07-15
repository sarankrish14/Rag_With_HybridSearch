"""Load and resolve RAG evaluation cases for offline / Streamlit eval runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from retrieval.pipeline import EvalCase, RetrieverConfig

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_EVAL_PATH = EVAL_DIR / "eval_cases.json"
EXAMPLE_EVAL_PATH = EVAL_DIR / "eval_cases.example.json"


def _resolve_chunk_ids_by_text(config: RetrieverConfig, snippets: list[str]) -> list[str]:
    if not snippets:
        return []

    from db.postgres_store import PostgresStore

    store = PostgresStore(config.database_url)
    found: list[str] = []
    try:
        for snippet in snippets:
            snippet = (snippet or "").strip()
            if not snippet:
                continue
            rows = store.fetch_all(
                """
                SELECT c.chunk_id
                FROM chunks c
                WHERE c.text ILIKE %s
                LIMIT 20
                """,
                (f"%{snippet}%",),
            )
            found.extend(row["chunk_id"] for row in rows if row.get("chunk_id"))
    finally:
        store.close()

    # Preserve order, drop duplicates.
    return list(dict.fromkeys(found))


def _normalize_case(raw: dict[str, Any], config: RetrieverConfig) -> EvalCase:
    question = str(raw.get("question", "")).strip()
    if not question:
        raise ValueError("Each eval case must include a non-empty 'question'.")

    keywords = [str(k).strip() for k in raw.get("expected_answer_keywords", []) if str(k).strip()]
    chunk_ids = [str(cid).strip() for cid in raw.get("expected_chunk_ids", []) if str(cid).strip()]

    if not chunk_ids:
        hints = raw.get("expected_chunk_text") or raw.get("expected_text_in_chunks") or []
        if isinstance(hints, str):
            hints = [hints]
        chunk_ids = _resolve_chunk_ids_by_text(config, list(hints))

    return EvalCase(
        question=question,
        expected_chunk_ids=chunk_ids,
        expected_answer_keywords=keywords,
    )


def load_eval_cases_from_data(
    data: dict[str, Any] | list[dict[str, Any]],
    config: RetrieverConfig,
) -> list[EvalCase]:
    """Parse eval JSON (list of cases or {\"cases\": [...]})."""
    if isinstance(data, list):
        raw_cases = data
    else:
        raw_cases = data.get("cases", [])
    if not raw_cases:
        raise ValueError("No eval cases found in JSON.")

    return [_normalize_case(case, config) for case in raw_cases]


def load_eval_cases_from_path(path: Path, config: RetrieverConfig) -> list[EvalCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return load_eval_cases_from_data(payload, config)


def default_eval_path() -> Path:
    return DEFAULT_EVAL_PATH if DEFAULT_EVAL_PATH.exists() else EXAMPLE_EVAL_PATH
