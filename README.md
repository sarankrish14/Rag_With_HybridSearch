# Rag_With_HybridSearch

RAG pipeline with **semantic chunking**, **hybrid search** (dense + BM25), **cross-encoder reranking**, Groq answers, a **Streamlit UI**, and a **FastAPI** HTTP API.

| Surface | How to start | Default URL |
|---------|--------------|-------------|
| Streamlit chat UI | `streamlit run streamlit_app.py` | http://localhost:8501 |
| FastAPI retrieval API | `python -m retrieval --serve` | http://localhost:8001 · docs: http://localhost:8001/docs |
| CLI query | `python -m retrieval "your question"` | — |

---

## Project layout

```
Rag_With_HybridSearch/
├── streamlit_app.py       # Chat UI + PDF ingest + evaluations
├── config.py              # Non-secret defaults (models, API host/port)
├── settings.py            # Loads secrets from .env
├── chromastore.py         # Optional ChromaDB mirror
├── db/                    # PostgreSQL + pgvector
├── models/                # Embedding model
├── semantic_chunker/      # PDF → chunks → embeddings → DB
├── retrieval/             # Hybrid search, rerank, Groq, FastAPI
├── documents/             # Sample / uploaded PDFs
├── evals/                 # Eval case loader + example JSON
├── tests/
├── requirements.txt
├── .env.example           # Template — copy to .env (never commit .env)
└── pyproject.toml
```

---

## Prerequisites

- Python **3.10+**
- PostgreSQL with [pgvector](https://github.com/pgvector/pgvector)
- [Groq API key](https://console.groq.com/)
- Optional: Redis (for query caching in Streamlit / API)

---

## 1. Setup

```powershell
cd d:\Rag_pipeline_Semantic_MyGit

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

copy .env.example .env
# Edit .env with your real Postgres, Groq, and RAG_API_KEY values

python -c "import nltk; nltk.download('punkt_tab')"
```

### Create the database

```sql
CREATE DATABASE rag_db;
\c rag_db
CREATE EXTENSION vector;
```

### `.env` variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `GROQ_API_KEY` | Yes | LLM answer generation |
| `RAG_API_KEY` or `API_KEY` | Yes | Protects FastAPI (`X-API-Key` header) |
| `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD` | Yes* | PostgreSQL connection |
| `DATABASE_URL` | Yes* | Alternative to `POSTGRES_*` |
| `GROQ_MODEL` | No | Default `llama-3.3-70b-versatile` |
| `RAG_REDIS_HOST` / `RAG_REDIS_PORT` | No | Redis cache (default localhost:6379) |

\* Set either `DATABASE_URL` **or** the `POSTGRES_*` fields.

**Do not commit `.env`.** It is listed in `.gitignore`.

---

## 2. Ingest a PDF (required before asking questions)

Chunks must exist in PostgreSQL before retrieval or Streamlit chat will work well.

```powershell
# From project root (venv activated)
python -m semantic_chunker documents/sample.pdf

# Re-process an already-ingested file
python -m semantic_chunker documents/sample.pdf --force

# Unit tests for the chunker
python -m semantic_chunker --test
```

You can also ingest from the **Streamlit sidebar** (Upload PDF → Ingest PDF).

---

## 3. Run Streamlit (recommended UI)

```powershell
cd d:\Rag_pipeline_Semantic_MyGit
.\.venv\Scripts\Activate.ps1
streamlit run streamlit_app.py
```

Open **http://localhost:8501** in your browser.

First launch may take **1–2 minutes** while the embedding model, reranker, and BM25 index load.

### What you can do in the UI

**Sidebar — Ingestion**
1. Upload a PDF
2. Optionally enable **Force re-ingest duplicates**
3. Click **Ingest PDF** → text is extracted, chunked, embedded, and saved to PostgreSQL
4. Search index reloads automatically after a successful ingest

**Sidebar — Retrieval settings**
| Control | Meaning |
|---------|---------|
| Cross-encoder rerank | Re-score top hits for better ranking (on by default) |
| Use Redis cache | Cache answers (leave off if Redis is not running) |
| Scope | Limit search to one ingested file, or all documents |
| Clear chat | Reset conversation history |
| Reload search index | Rebuild BM25 / reload pipeline after external DB changes |

**Tab: Chat**
- Type a question in the chat input
- Answer shows with expandable **Sources**, **Retrieved chunks**, and **Latency**

**Tab: Evaluations**
- Run Recall@K / Precision@K / MRR / NDCG / answer coverage
- Upload, paste, or use `evals/eval_cases.example.json`

Example eval JSON:

```json
{
  "cases": [
    {
      "question": "What is semantic chunking?",
      "expected_answer_keywords": ["semantic", "chunk"],
      "expected_chunk_text": ["semantic chunking"]
    }
  ]
}
```

---

## 4. Run the FastAPI retrieval API

```powershell
cd d:\Rag_pipeline_Semantic_MyGit
.\.venv\Scripts\Activate.ps1
python -m retrieval --serve
```

| Item | Value |
|------|--------|
| Base URL | `http://localhost:8001` |
| Interactive docs (Swagger) | http://localhost:8001/docs |
| Host / port | `0.0.0.0:8001` (from `config.py`) |
| Auth header (all endpoints) | `X-API-Key: <same value as RAG_API_KEY in .env>` |
| Rate limits | `/query` 10/min · `/health` & `/stats` 30/min · `/cache/flush` 5/min · default 60/min |

### Endpoints overview

| Method | Path | Body | Purpose |
|--------|------|------|---------|
| `POST` | `/query` | JSON `QueryRequest` | Full RAG: search + answer |
| `GET` | `/health` | — | Postgres / Redis / BM25 / embedding status |
| `GET` | `/stats` | — | Query count, cache hit rate, avg latency |
| `POST` | `/cache/flush` | — | Clear Redis query cache |

---

### `POST /query` — request payload

```http
POST http://localhost:8001/query
Content-Type: application/json
X-API-Key: your-rag-api-key-from-env
```

**Body fields**

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| `query` | string | Yes | — | Length 1–2000 characters |
| `metadata_filters` | object or `null` | No | `null` | Allowed keys only (see below) |
| `rerank` | boolean | No | `true` | Use cross-encoder reranking |

**Allowed `metadata_filters` keys:** `source_file`, `doc_id`, `page_number`, `section`

**Minimal example**

```json
{
  "query": "What is semantic chunking?"
}
```

**Full example (filter to one PDF, keep rerank)**

```json
{
  "query": "What is semantic chunking?",
  "metadata_filters": {
    "source_file": "sample.pdf"
  },
  "rerank": true
}
```

**Filter by page**

```json
{
  "query": "Summarize this page",
  "metadata_filters": {
    "source_file": "sample.pdf",
    "page_number": 3
  },
  "rerank": true
}
```

**PowerShell (Invoke-RestMethod)**

```powershell
$headers = @{
  "X-API-Key" = "your-rag-api-key-from-env"
  "Content-Type" = "application/json"
}
$body = @{
  query = "What is semantic chunking?"
  metadata_filters = @{ source_file = "sample.pdf" }
  rerank = $true
} | ConvertTo-Json

Invoke-RestMethod -Method POST -Uri "http://localhost:8001/query" -Headers $headers -Body $body
```

**curl**

```bash
curl -X POST "http://localhost:8001/query" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-rag-api-key-from-env" \
  -d "{\"query\":\"What is semantic chunking?\",\"metadata_filters\":{\"source_file\":\"sample.pdf\"},\"rerank\":true}"
```

---

### `POST /query` — response payload

```json
{
  "answer": "Semantic chunking splits text by meaning rather than fixed token size.",
  "citations": [
    {
      "source_file": "sample.pdf",
      "page_number": 3,
      "chunk_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    }
  ],
  "has_answer": true,
  "is_hallucination_risk": false,
  "chunks_used": [
    {
      "chunk_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
      "source_file": "sample.pdf",
      "page_number": 3,
      "section": "Introduction",
      "text": "…chunk text…",
      "final_rank": 1,
      "rerank_score": 0.87
    }
  ],
  "llm_context_chunks": [
    {
      "chunk_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
      "source_file": "sample.pdf",
      "page_number": 3,
      "section": "Introduction",
      "final_rank": 1,
      "neighbors_added": 1,
      "expanded_text": "…chunk plus neighbor context…"
    }
  ],
  "latency_breakdown": {
    "preprocess_ms": 2.1,
    "embed_ms": 15.0,
    "retrieve_ms": 40.0,
    "rerank_ms": 25.0,
    "llm_ms": 800.0,
    "total_ms": 900.0
  },
  "cached": false,
  "query_type": "question"
}
```

**Response fields**

| Field | Meaning |
|-------|---------|
| `answer` | Final natural-language answer |
| `citations` | Sources the model cited (`source_file`, `page_number`, `chunk_id`) |
| `has_answer` | `false` if the model reported no answer in context |
| `is_hallucination_risk` | `true` if citations do not match retrieved context |
| `chunks_used` | Ranked chunks that drove the answer |
| `llm_context_chunks` | Expanded text (with neighbors) sent to Groq |
| `latency_breakdown` | Per-step timings in milliseconds |
| `cached` | `true` if served from Redis |
| `query_type` | e.g. `question`, `greeting`, `empty`, `too_short`, `too_long` |

Response also includes header `X-Request-ID` for log tracing.

**Auth error**

```json
{ "detail": "Invalid API key" }
```
HTTP status **403** if `X-API-Key` is missing or wrong.

---

### `GET /health`

```powershell
Invoke-RestMethod -Uri "http://localhost:8001/health" -Headers @{ "X-API-Key" = "your-rag-api-key-from-env" }
```

Example response:

```json
{
  "status": "ok",
  "postgres": "connected",
  "redis": "disconnected",
  "bm25_index": "loaded",
  "embedding_model": "loaded"
}
```

---

### `GET /stats`

```json
{
  "total_queries": 12,
  "cache_hit_rate": 0.25,
  "average_latency_per_step_ms": {
    "total_ms": 950.0,
    "llm_ms": 800.0
  },
  "json_parse_failure_count": 0
}
```

---

### `POST /cache/flush`

Call after re-ingesting documents so old cached answers are cleared.

```json
{ "flushed_keys": 3 }
```

---

## 5. CLI retrieval (no UI / no server)

```powershell
# Demo questions
python -m retrieval --demo

# One question
python -m retrieval "What is semantic chunking?"
```

---

## 6. Python API (in-process)

```python
from semantic_chunker import semantic_chunk_pdf
from retrieval import RAGPipeline, load_config

# Ingest
chunks = semantic_chunk_pdf("documents/sample.pdf")

# Query (sync — used by Streamlit)
pipeline = RAGPipeline(load_config())
try:
    result = pipeline.run(
        "What is this document about?",
        metadata_filters={"source_file": "sample.pdf"},  # optional
        rerank=True,
        use_cache=False,
    )
    print(result.answer)
    print(result.citations)
    print(result.latency_breakdown)
finally:
    pipeline.close()
```

Async (used by FastAPI):

```python
import asyncio
from retrieval import RAGPipeline, load_config

async def main():
    pipeline = await RAGPipeline.create(load_config())
    try:
        result = await pipeline.arun("What is semantic chunking?", use_cache=False)
        print(result.answer)
    finally:
        await pipeline.aclose()

asyncio.run(main())
```

---

## 7. How the pipeline works (short)

1. **Preprocess** query (greetings / empty queries skip search)
2. **Embed** the question
3. **Hybrid retrieve** — dense (pgvector) + sparse (BM25), fused with RRF
4. **Optional metadata filter** (`source_file`, `page_number`, …)
5. **Rerank** with a cross-encoder
6. **Expand** with neighbor chunks + token budget
7. **Groq LLM** returns JSON answer + citations
8. **Validate** citations against retrieved context

Deeper file-by-file detail: [retrieval/README.md](retrieval/README.md).

---

## 8. Tests

```powershell
pytest
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Streamlit / API can't connect to DB | Check `POSTGRES_*` or `DATABASE_URL` in `.env`; confirm pgvector is installed |
| `GROQ_API_KEY is not set` | Add key to `.env`, restart the process |
| API returns 403 | Send header `X-API-Key` matching `RAG_API_KEY` |
| No useful answers | Ingest a PDF first (`python -m semantic_chunker …` or Streamlit upload) |
| Redis timeout in Streamlit | Turn off **Use Redis cache** in the sidebar |
| Slow first query | Normal — models and BM25 load on first startup |
| Stale answers after re-ingest | `POST /cache/flush` or disable cache |

---

## Quick start checklist

1. Copy `.env.example` → `.env` and fill secrets  
2. Create Postgres DB + `CREATE EXTENSION vector;`  
3. `pip install -r requirements.txt`  
4. `python -m semantic_chunker documents/sample.pdf`  
5. UI: `streamlit run streamlit_app.py` → http://localhost:8501  
6. API: `python -m retrieval --serve` → http://localhost:8001/docs with `X-API-Key`
