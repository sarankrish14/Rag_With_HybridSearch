"""
Streamlit chat UI for RagPipelines — PDF ingestion + RAG retrieval + evals.

Run from the project root:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

from evals.loader import (
    default_eval_path,
    load_eval_cases_from_data,
    load_eval_cases_from_path,
)
from retrieval import RAGPipeline, load_config
from retrieval.pipeline import (
    EvalReport,
    HybridRetriever,
    PipelineEvaluator,
    RetrieverConfig,
)
from semantic_chunker import semantic_chunk_pdf

PROJECT_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = PROJECT_ROOT / "documents" / "uploads"


def _invalidate_bm25_cache(config: RetrieverConfig) -> None:
    cache_path = HybridRetriever._bm25_cache_path(config)
    if cache_path.exists():
        cache_path.unlink()


def _list_source_files(config: RetrieverConfig) -> list[str]:
    from db.postgres_store import PostgresStore

    store = PostgresStore(config.database_url)
    try:
        rows = store.fetch_all(
            "SELECT file_name FROM documents ORDER BY created_at DESC"
        )
        return [row["file_name"] for row in rows if row.get("file_name")]
    except Exception:
        return []
    finally:
        store.close()


@st.cache_resource(show_spinner="Loading embedding model, reranker, and search index…")
def load_pipeline() -> RAGPipeline:
    return RAGPipeline(load_config())


def reload_pipeline() -> RAGPipeline:
    cached = load_pipeline()
    try:
        cached.close()
    except Exception:
        pass
    load_pipeline.clear()
    return load_pipeline()


def ingest_pdf(file_path: Path, *, force: bool) -> list[dict]:
    return semantic_chunk_pdf(str(file_path), force=force)


def run_query(
    pipeline: RAGPipeline,
    question: str,
    *,
    metadata_filters: Optional[dict[str, Any]],
    rerank: bool,
    use_cache: bool,
):
    return pipeline.run(
        question,
        metadata_filters=metadata_filters,
        rerank=rerank,
        use_cache=use_cache,
    )


def _init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "eval_report" not in st.session_state:
        st.session_state.eval_report = None


def _render_assistant_message(msg: dict[str, Any]) -> None:
    st.markdown(msg["content"])

    if msg.get("is_hallucination_risk"):
        st.warning("Citation validation flagged a possible hallucination risk.")

    citations = msg.get("citations") or []
    if citations:
        with st.expander("Sources"):
            for cite in citations:
                source = cite.get("source_file") or cite.get("source", "unknown")
                page = cite.get("page_number") or cite.get("page")
                page_text = f", page {page}" if page is not None else ""
                st.markdown(f"- **{source}**{page_text}")

    chunks_used = msg.get("chunks_used") or []
    if chunks_used:
        with st.expander("Retrieved chunks"):
            for chunk in chunks_used:
                source = chunk.get("metadata", {}).get("source_file", "unknown")
                page = chunk.get("metadata", {}).get("page_number", "?")
                rank = chunk.get("final_rank", "?")
                preview = (chunk.get("text") or "")[:280]
                st.markdown(f"**Rank {rank}** · {source} · p.{page}")
                st.caption(preview + ("…" if len(chunk.get("text") or "") > 280 else ""))

    latency = msg.get("latency_ms")
    if latency is not None:
        with st.expander("Latency"):
            breakdown = msg.get("latency_breakdown") or {}
            st.metric("Total", f"{latency:.0f} ms")
            if breakdown:
                cols = st.columns(3)
                labels = [
                    ("preprocess_ms", "Preprocess"),
                    ("retrieve_ms", "Retrieve"),
                    ("llm_ms", "LLM"),
                ]
                for col, (key, label) in zip(cols, labels):
                    if key in breakdown:
                        col.caption(f"{label}: {breakdown[key]:.0f} ms")


def _score_color(value: float, *, good: float = 0.7, ok: float = 0.4) -> str:
    if value >= good:
        return "🟢"
    if value >= ok:
        return "🟡"
    return "🔴"


def _render_eval_report(report: EvalReport) -> None:
    st.subheader("Evaluation summary")
    st.caption(f"{report.total_cases} cases · {report.eval_time_ms:.0f} ms total")

    cols = st.columns(5)
    metrics = [
        ("Recall@K", report.avg_recall_at_k),
        ("Precision@K", report.avg_precision_at_k),
        ("MRR", report.avg_mrr),
        ("NDCG@K", report.avg_ndcg_at_k),
        ("Answer coverage", report.avg_answer_coverage),
    ]
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, f"{value:.1%}", help=f"{_score_color(value)} quality indicator")

    rows = []
    for i, result in enumerate(report.cases, start=1):
        rows.append(
            {
                "#": i,
                "Question": result.case.question,
                "Recall@K": result.recall_at_k,
                "Precision@K": result.precision_at_k,
                "MRR": result.mrr,
                "NDCG@K": result.ndcg_at_k,
                "Answer coverage": result.answer_coverage,
                "Expected chunks": len(result.case.expected_chunk_ids),
                "Retrieved top-K": len(result.retrieved_ids),
            }
        )

    st.subheader("Per-case results")
    df = pd.DataFrame(rows)
    display_df = df.copy()
    for col in ("Recall@K", "Precision@K", "Answer coverage"):
        display_df[col] = display_df[col].map(lambda v: f"{v:.1%}")
    for col in ("MRR", "NDCG@K"):
        display_df[col] = display_df[col].map(lambda v: f"{v:.3f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.subheader("Metric comparison")
    chart_df = pd.DataFrame(
        {
            "Metric": [m[0] for m in metrics],
            "Score": [m[1] for m in metrics],
        }
    ).set_index("Metric")
    st.bar_chart(chart_df)

    with st.expander("Raw eval details (JSON)"):
        payload = {
            "summary": {
                "total_cases": report.total_cases,
                "eval_time_ms": report.eval_time_ms,
                "avg_recall_at_k": report.avg_recall_at_k,
                "avg_precision_at_k": report.avg_precision_at_k,
                "avg_mrr": report.avg_mrr,
                "avg_ndcg_at_k": report.avg_ndcg_at_k,
                "avg_answer_coverage": report.avg_answer_coverage,
            },
            "cases": [
                {
                    "question": r.case.question,
                    "expected_chunk_ids": r.case.expected_chunk_ids,
                    "expected_answer_keywords": r.case.expected_answer_keywords,
                    "retrieved_ids": r.retrieved_ids,
                    "recall_at_k": r.recall_at_k,
                    "precision_at_k": r.precision_at_k,
                    "mrr": r.mrr,
                    "ndcg_at_k": r.ndcg_at_k,
                    "answer_coverage": r.answer_coverage,
                    "faithful": r.faithful,
                }
                for r in report.cases
            ],
        }
        st.json(payload)


def _render_eval_tab(pipeline: RAGPipeline, config: RetrieverConfig) -> None:
    st.subheader("RAG evaluation")
    st.caption(
        "Measure retrieval quality (Recall, Precision, MRR, NDCG) and answer keyword coverage."
    )

    col_left, col_right = st.columns([1, 1])
    with col_left:
        eval_k = st.slider("K (top chunks)", min_value=3, max_value=15, value=5)
        uploaded_eval = st.file_uploader(
            "Upload eval JSON",
            type=["json"],
            help="Use evals/eval_cases.example.json as a template.",
        )
    with col_right:
        default_path = default_eval_path()
        use_default = st.checkbox(
            f"Include bundled cases ({default_path.name})",
            value=not uploaded_eval,
        )
        st.text_area(
            "Or paste eval JSON",
            height=160,
            placeholder='{"cases": [{"question": "...", "expected_answer_keywords": ["..."], "expected_chunk_text": ["..."]}]}',
            key="eval_json_paste",
        )

    with st.expander("Eval JSON format"):
        st.markdown(
            """
Each case supports:
- **question** — query to run
- **expected_answer_keywords** — keywords that should appear in the generated answer
- **expected_chunk_ids** — optional exact chunk UUIDs
- **expected_chunk_text** — optional text snippets; matching chunk IDs are resolved from the DB
            """
        )
        if default_path.exists():
            st.code(default_path.read_text(encoding="utf-8"), language="json")

    run_eval = st.button("Run evaluation", type="primary", use_container_width=True)

    if run_eval:
        cases = []
        errors: list[str] = []

        if use_default and default_path.exists():
            try:
                cases.extend(load_eval_cases_from_path(default_path, config))
            except Exception as exc:
                errors.append(f"Bundled cases: {exc}")

        if uploaded_eval is not None:
            try:
                data = json.loads(uploaded_eval.getvalue().decode("utf-8"))
                cases.extend(load_eval_cases_from_data(data, config))
            except Exception as exc:
                errors.append(f"Uploaded file: {exc}")

        pasted = (st.session_state.get("eval_json_paste") or "").strip()
        if pasted:
            try:
                data = json.loads(pasted)
                cases.extend(load_eval_cases_from_data(data, config))
            except Exception as exc:
                errors.append(f"Pasted JSON: {exc}")

        if errors:
            for err in errors:
                st.error(err)

        if not cases:
            st.warning("Add at least one eval case via upload, paste, or bundled file.")
        else:
            with st.spinner(f"Running {len(cases)} eval cases (calls Groq per case)…"):
                try:
                    evaluator = PipelineEvaluator(pipeline)
                    report = evaluator.evaluate(cases, k=eval_k)
                except Exception as exc:
                    st.error(f"Evaluation failed: {exc}")
                else:
                    st.session_state.eval_report = report
                    st.success("Evaluation complete.")

    if st.session_state.eval_report is not None:
        report = st.session_state.eval_report
        _render_eval_report(report)

        st.download_button(
            "Download results JSON",
            data=json.dumps(
                {
                    "summary": {
                        "total_cases": report.total_cases,
                        "eval_time_ms": report.eval_time_ms,
                        "avg_recall_at_k": report.avg_recall_at_k,
                        "avg_precision_at_k": report.avg_precision_at_k,
                        "avg_mrr": report.avg_mrr,
                        "avg_ndcg_at_k": report.avg_ndcg_at_k,
                        "avg_answer_coverage": report.avg_answer_coverage,
                    },
                    "cases": [
                        {
                            "question": r.case.question,
                            "expected_chunk_ids": r.case.expected_chunk_ids,
                            "expected_answer_keywords": r.case.expected_answer_keywords,
                            "retrieved_ids": r.retrieved_ids,
                            "recall_at_k": r.recall_at_k,
                            "precision_at_k": r.precision_at_k,
                            "mrr": r.mrr,
                            "ndcg_at_k": r.ndcg_at_k,
                            "answer_coverage": r.answer_coverage,
                        }
                        for r in report.cases
                    ],
                },
                indent=2,
            ),
            file_name="rag_eval_results.json",
            mime="application/json",
            use_container_width=True,
        )


def _render_chat_tab(
    pipeline: RAGPipeline,
    *,
    metadata_filters: Optional[dict[str, Any]],
    rerank: bool,
    use_cache: bool,
) -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and "latency_ms" in msg:
                _render_assistant_message(msg)
            else:
                st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about your documents…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer…"):
                try:
                    result = run_query(
                        pipeline,
                        prompt,
                        metadata_filters=metadata_filters,
                        rerank=rerank,
                        use_cache=use_cache,
                    )
                except Exception as exc:
                    st.error(f"Query failed: {exc}")
                else:
                    assistant_msg = {
                        "role": "assistant",
                        "content": result.answer,
                        "citations": [asdict(c) for c in result.citations],
                        "chunks_used": [asdict(c) for c in result.chunks_used],
                        "has_answer": result.has_answer,
                        "is_hallucination_risk": result.is_hallucination_risk,
                        "latency_ms": result.latency_breakdown.get("total_ms"),
                        "latency_breakdown": result.latency_breakdown,
                    }
                    _render_assistant_message(assistant_msg)
                    st.session_state.messages.append(assistant_msg)


def main() -> None:
    st.set_page_config(
        page_title="RAG Chat",
        page_icon="📚",
        layout="wide",
    )

    _init_session_state()
    config = load_config()

    st.title("RAG Document Chat")
    st.caption(
        "Upload PDFs to ingest them, then ask questions grounded in your documents."
    )

    with st.sidebar:
        st.header("Ingestion")
        uploaded = st.file_uploader("Upload PDF", type=["pdf"])
        force_reingest = st.checkbox("Force re-ingest duplicates", value=False)

        if st.button("Ingest PDF", use_container_width=True, disabled=uploaded is None):
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            dest = UPLOAD_DIR / uploaded.name
            dest.write_bytes(uploaded.getvalue())

            with st.status("Running ingestion pipeline…", expanded=True) as status:
                st.write("Extracting text, chunking, embedding, and saving to PostgreSQL…")
                try:
                    chunks = ingest_pdf(dest, force=force_reingest)
                except Exception as exc:
                    status.update(label="Ingestion failed", state="error")
                    st.error(str(exc))
                else:
                    _invalidate_bm25_cache(config)
                    reload_pipeline()
                    if chunks:
                        status.update(
                            label=f"Ingested {len(chunks)} chunks from {uploaded.name}",
                            state="complete",
                        )
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": (
                                    f"Successfully ingested **{uploaded.name}** "
                                    f"({len(chunks)} chunks). You can ask questions about it now."
                                ),
                            }
                        )
                    else:
                        status.update(
                            label="Already processed (skipped)",
                            state="complete",
                        )
                        st.info(
                            "This file was already ingested. "
                            "Enable **Force re-ingest** to process it again."
                        )
            st.rerun()

        st.divider()
        st.header("Retrieval settings")
        rerank = st.toggle("Cross-encoder rerank", value=True)
        use_cache = st.toggle(
            "Use Redis cache",
            value=False,
            help="Requires a running Redis server. Leave off if you see Redis timeout warnings.",
        )

        source_files = _list_source_files(config)
        selected_source: Optional[str] = None
        if source_files:
            options = ["All documents"] + source_files
            pick = st.selectbox("Scope", options)
            if pick != "All documents":
                selected_source = pick
        else:
            st.caption("No documents in the database yet.")

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        if st.button("Reload search index", use_container_width=True):
            _invalidate_bm25_cache(config)
            reload_pipeline()
            st.success("Search index reloaded.")

    metadata_filters = (
        {"source_file": selected_source} if selected_source else None
    )

    with st.spinner("Loading RAG models and search index (first run may take 1–2 minutes)…"):
        try:
            pipeline = load_pipeline()
        except Exception as exc:
            st.error(f"Failed to start retrieval pipeline: {exc}")
            st.info(
                "Check `.env` for `DATABASE_URL`, `GROQ_API_KEY`, and PostgreSQL/pgvector setup."
            )
            return

    tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Evaluations"])

    with tab_chat:
        _render_chat_tab(
            pipeline,
            metadata_filters=metadata_filters,
            rerank=rerank,
            use_cache=use_cache,
        )

    with tab_eval:
        _render_eval_tab(pipeline, config)


if __name__ == "__main__":
    main()
