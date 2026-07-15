# Retrieval Pipeline — Beginner's Guide

This folder is the **retrieval** half of RAG (Retrieval-Augmented Generation).

**Simple idea:** You ask a question → the system finds relevant text from your uploaded PDFs → an AI (Groq) reads that text and writes an answer with citations.

You must run **ingestion** first (`python -m semantic_chunker documents/sample.pdf`) so PostgreSQL has chunks to search.

---

## What happens when you ask a question?

Think of it like a library research assistant:

1. **Clean your question** — remove junk, block "hi" from wasting AI money
2. **Turn question into numbers** — an embedding (vector) that captures meaning
3. **Search the database two ways** — meaning search + keyword search, then merge results
4. **Filter** — optionally keep only chunks from a specific file or page
5. **Rerank** — a smarter model re-scores the best matches
6. **Add neighbor text** — grab chunks before/after so context is not cut mid-sentence
7. **Build the prompt** — pack context + question for the AI, respecting token limits
8. **Call Groq** — the LLM writes a JSON answer
9. **Validate** — check citations and parse the response
10. **Return** — final answer, sources, and timing info

```mermaid
flowchart LR
    Q[Your question] --> S1[Step 1: Preprocess]
    S1 --> S2[Step 2: Embed]
    S2 --> S3[Step 3: Hybrid search]
    S3 --> S4[Step 4: Filter]
    S4 --> S5[Step 5: Rerank]
    S5 --> S6[Step 6: Expand context]
    S6 --> S7[Step 7: Build prompt]
    S7 --> S8[Step 8: Groq LLM]
    S8 --> S9[Step 9: Process response]
    S9 --> A[Answer + citations]
```

---

## Folder layout

| File | What it does (one sentence) |
|------|-------------------------------|
| `__init__.py` | Public imports — what other code should use from this package |
| `__main__.py` | Command-line entry: run queries or start the web server |
| `pipeline.py` | The main brain — all 9 steps, config, data types, API server |
| `token_budget.py` | Counts tokens and trims text so prompts fit the LLM window |
| `groq_async.py` | Sends HTTP requests to Groq's chat API |
| `auth.py` | Checks API key on protected web endpoints |

---

## How to run

From the `RagPipelines` folder:

```powershell
# Demo with 3 sample questions
python -m retrieval --demo

# Ask one question
python -m retrieval "What is semantic chunking?"

# Start web API (Swagger UI at http://localhost:8001/docs)
python -m retrieval --serve
```

**Python code example:**

```python
import asyncio
from retrieval import RAGPipeline, load_config

async def main():
    pipeline = await RAGPipeline.create(load_config())
    try:
        result = await pipeline.arun("What is semantic chunking?")
        print(result.answer)
        print(result.citations)
    finally:
        await pipeline.aclose()

asyncio.run(main())
```

---

## Key concepts (plain English)

| Term | Meaning | Example |
|------|---------|---------|
| **Chunk** | A small piece of text cut from a PDF | One paragraph about "semantic chunking" |
| **Embedding** | A list of numbers representing meaning | `[0.12, -0.05, 0.88, ...]` |
| **Dense search** | Find chunks with similar *meaning* (pgvector) | "automobile" matches "car" |
| **Sparse search (BM25)** | Find chunks with matching *words* | Good for exact codes like "ISO-9001" |
| **RRF fusion** | Merge two ranked lists into one fair score | Chunk #3 in both lists rises to top |
| **Reranker** | A slower, more accurate second pass | Reads question + chunk together |
| **Token** | Roughly a word piece the LLM counts | "chunking" might be 2 tokens |
| **Citation** | Where the answer came from | `sample.pdf`, page 5 |

---

# File-by-file explanation

## `__main__.py` — CLI runner

This file runs when you type `python -m retrieval`.

| Function | What it does |
|----------|--------------|
| `main()` | Reads command-line args and decides: server mode, demo mode, or single query |
| `_run_queries(queries)` | Creates the pipeline, runs each question, prints the answer |

**Example flow:**

```
python -m retrieval "What is RAG?"
```

1. `main()` sees no `--serve` or `--demo`
2. Builds `test_queries = ["What is RAG?"]`
3. `asyncio.run(_run_queries(...))` runs the async pipeline
4. Prints `result.answer`

**`--serve`:** starts FastAPI with uvicorn on `config.API_HOST` / `config.API_PORT`.

**`--demo`:** runs three hard-coded test questions.

---

## `__init__.py` — Public API

Exports the symbols other projects should import:

| Name | What it is |
|------|------------|
| `RAGPipeline` | Main class — run the full pipeline |
| `RetrieverConfig` | Settings object (top_k, model names, etc.) |
| `load_config()` | Builds `RetrieverConfig` from `.env` |
| `create_rag_pipeline()` | Shortcut: `RAGPipeline(load_config())` |
| `PipelineResult` | What you get back after `run()` |
| `ExpandedChunk` | A chunk plus neighbor text sent to the LLM |
| `format_llm_context_for_display()` | Pretty-print what was sent to Groq |

---

## `auth.py` — API key check

Used only by the FastAPI server.

| Name | What it does |
|------|--------------|
| `API_KEY_HEADER` | Expects header `X-API-Key: your-secret-key` |
| `verify_api_key()` | Compares header to `settings.api_key`; returns 403 if wrong |

**Example request:**

```http
POST /query
X-API-Key: your-api-key-from-env
Content-Type: application/json

{"query": "What is semantic chunking?"}
```

---

## `groq_async.py` — Groq HTTP client

Talks to Groq's cloud API (like OpenAI chat completions).

| Class / function | What it does |
|----------------|--------------|
| `GroqAPIError` | Custom error with HTTP status (used for retries on 429/503) |
| `AsyncGroqClient` | Holds an `httpx` client with your API key |
| `AsyncGroqClient.close()` | Closes the HTTP connection pool |
| `AsyncGroqClient.chat_completion()` | POST to `/chat/completions` with messages + optional JSON mode |

**Example payload (simplified):**

```json
{
  "model": "llama-3.3-70b-versatile",
  "messages": [
    {"role": "system", "content": "Answer using only context..."},
    {"role": "user", "content": "CONTEXT:\n...\n\nQuestion: What is RAG?"}
  ],
  "response_format": {"type": "json_object"}
}
```

---

## `token_budget.py` — Fit text inside LLM limits

LLMs have a maximum context window (e.g. 128k tokens). This module counts and trims.

| Function | What it does | Example |
|----------|--------------|---------|
| `count_tokens(text)` | How many tokens in a string | `count_tokens("hello")` → small number |
| `count_messages_tokens(system, user)` | System + user + overhead | Used before calling Groq |
| `max_input_tokens(window, max_output, reserve)` | Safe input budget | `128000 - 1024 - 64` |
| `clip_text_to_tokens(text, max)` | Cut text without breaking encoding | Long chunk → truncated |
| `count_chunk_list_tokens(chunks)` | Total tokens in all context chunks | Sum of expanded texts |
| `trim_chunks_to_budget(chunks, budget)` | Drop lowest-priority chunks until under budget | Removes rank-5 before rank-1 |
| `build_user_prompt(question, blocks)` | Formats `CONTEXT:` + question | Standard user message shape |
| `fit_prompt_to_budget(...)` | Main function: keep as many chunks as fit | Returns kept chunks + final prompt |

**Beginner analogy:** Like packing a suitcase — you add clothes (chunks) until the bag (token limit) is full, then remove the least important items first.

---

# `pipeline.py` — The main pipeline

This is the largest file. Below is every **class** and **important function**, grouped by step.

---

## Helper utilities (top of file)

| Name | Purpose |
|------|---------|
| `_RequestIDFilter` | Adds `req=abc123` to log lines so you can trace one request |
| `_get_logger(name)` | Creates a logger like `rag.retrieval.pipeline` |
| `_ensure_request_id()` | Generates a short UUID for logging |
| `_observe_pipeline_metrics(breakdown)` | Sends timing to Prometheus (if installed) |
| `_utc_iso()` | Current time as ISO string |
| `_ms_since(start)` | Milliseconds since `time.perf_counter()` |
| `_Secret` | Wraps API keys so logs print `***` not the real key |
| `_json_default` | JSON serializer helper for secrets |
| `_parse_json_field(value)` | Safely parse JSON metadata from DB rows |
| `_row_to_chunk_metadata(row)` | Turn a PostgreSQL row into a metadata dict |
| `_CHITCHAT_RESPONSES` | Friendly replies for "hi", empty query, etc. |

---

## Configuration

### `RetrieverConfig` (dataclass)

One object holding all tunable settings.

| Field | Meaning | Typical value |
|-------|---------|---------------|
| `database_url` | PostgreSQL connection string | from `.env` |
| `embedding_model_name` | Model for query vectors | same as ingestion |
| `n_dense` | How many vector search hits | e.g. 20 |
| `n_sparse` | How many BM25 hits | e.g. 20 |
| `rrf_k` | RRF smoothing constant | 60 |
| `top_k_after_fusion` | Chunks after merging | 30 |
| `final_top_k` | Chunks after reranking | e.g. 5 |
| `reranker_model_name` | Cross-encoder model | e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `neighbor_chunks` | Chunks before/after to include | 1 |
| `max_context_tokens` | Max context for LLM | 3000 |
| `llm_model` | Groq model name | `llama-3.3-70b-versatile` |
| `groq_api_key` | Groq secret | from `.env` |
| `redis_host` / `redis_port` | Optional cache | localhost:6379 |
| `min_query_length` / `max_query_length` | Query size limits | 3 / 2000 |

### `load_config()`

1. Starts with defaults from `config.py`
2. Overrides any `RAG_*` environment variable (e.g. `RAG_FINAL_TOP_K=3`)

---

## Data structures (what flows through the pipeline)

These are `@dataclass` objects — simple bags of named fields.

| Class | Holds |
|-------|-------|
| `QueryType` | Enum: `QUESTION`, `GREETING`, `EMPTY`, `TOO_SHORT`, `TOO_LONG`, `ACKNOWLEDGMENT` |
| `ProcessedQuery` | Cleaned text, type, `is_valid`, language |
| `EmbeddingResult` | Vector, cache hit?, timing |
| `RetrievedChunk` | One search hit with dense/sparse/RRF scores |
| `HybridRetrievalResult` | List of chunks + search timings |
| `FilterResult` | How many chunks removed by metadata filter |
| `RankedChunk` | Chunk after reranking with `final_rank` |
| `RerankerResult` | Ranked list + rerank time |
| `ExpandedChunk` | Core text + neighbors merged for LLM |
| `ContextBuilderResult` | All expanded chunks + token estimate |
| `BuiltPrompt` | `system_prompt` + `user_prompt` ready for Groq |
| `LLMResponse` | Raw JSON text + token usage from API |
| `Citation` | `source_file`, `page_number`, `chunk_id` |
| `ProcessedResponse` | Parsed answer + validated citations |
| `PipelineResult` | **Final output** — answer, citations, timings, chunks used |
| `EvalCase` / `EvalResult` / `EvalReport` | For testing retrieval quality |

### `format_llm_context_for_display(chunks)`

Debug helper — prints exactly what text was sent to Groq, with source/page headers.

---

## Step 1: `QueryPreprocessor`

**Job:** Clean and validate the user's text before any expensive work.

| Method | What it does |
|--------|--------------|
| `process(raw_query)` | Main entry — returns `ProcessedQuery` |
| `_clean(text)` | Normalize unicode, strip bad chars, collapse spaces |
| `_sanitize_prompt_injection(text)` | Remove patterns like "ignore previous instructions" |
| `_classify(cleaned)` | Detect greeting vs real question |
| `_detect_language(text)` | Optional language code via `langdetect` |
| `_reject(...)` | Build invalid `ProcessedQuery` with reason |

**Examples:**

| Input | Result |
|-------|--------|
| `"  What is RAG?  "` | Valid question, cleaned |
| `"hi"` | `QueryType.GREETING`, `is_valid=False` — no DB search |
| `""` | `QueryType.EMPTY` |
| `"ab"` | `QueryType.TOO_SHORT` if under `min_query_length` |

---

## Step 2: `QueryEmbedder`

**Job:** Convert the cleaned question into an embedding vector (same model as ingestion).

| Method | What it does |
|--------|--------------|
| `embed(processed_query)` | Returns `EmbeddingResult` |
| Internal `_cache` | LRU cache — repeated questions are instant |

**Example:**

```
Question: "What is semantic chunking?"
→ embedding: [0.023, -0.114, 0.556, ...]  (hundreds of floats)
```

---

## Step 3: `HybridRetriever`

**Job:** Find relevant chunks using **two search methods**, then **fuse** results.

### Dense search (pgvector)

Compares your query vector to stored chunk vectors in PostgreSQL. Good for meaning.

### Sparse search (BM25)

Keyword scoring over all chunk texts loaded into memory. Good for exact terms.

### RRF merge (`_rrf_merge`)

Formula per chunk: `score += 1 / (k + rank)` from each list.  
Chunks appearing in **both** lists get a boost.

| Method | What it does |
|--------|--------------|
| `retrieve(query, embedding)` | Sync: dense + sparse + fuse |
| `aretrieve(query, embedding)` | Async version for API |
| `_dense_search` / `_adense_search` | PostgreSQL vector search |
| `_sparse_search` | BM25 over in-memory index |
| `_rrf_merge` | Combine ranked lists |
| `create(config, store)` | Async factory — loads BM25 corpus |
| `_load_corpus_paged` | Load all chunks from DB (with disk cache in `bm25_cache/`) |

**Simple example:**

```
Query: "semantic chunking"

Dense top hit:  chunk_A (similar meaning)
Sparse top hit: chunk_B (contains words "semantic" and "chunking")
RRF winner:     chunk that ranks well in BOTH
```

---

## Step 4: `MetadataFilter`

**Job:** Keep only chunks matching optional filters (e.g. one PDF).

| Method | What it does |
|--------|--------------|
| `apply(chunks, filters)` | Returns filtered list + `FilterResult` |
| `_passes(metadata, filters)` | True if all filter keys match |

**Example filter:**

```python
metadata_filters = {"source_file": "sample.pdf"}
# Only chunks from sample.pdf survive
```

---

## Step 5: `Reranker`

**Job:** Re-score top candidates with a **cross-encoder** (reads question + chunk together).

| Method | What it does |
|--------|--------------|
| `rerank(query, chunks)` | Returns top `final_top_k` as `RankedChunk` list |
| On failure | Falls back to RRF order |

**Why?** First search is fast but rough. Reranking is slower but much more accurate.

---

## Step 6: `ContextBuilder`

**Job:** For each top chunk, fetch **neighbor chunks** from the same document and join their text.

| Method | What it does |
|--------|--------------|
| `expand(ranked_chunks)` | Sync expansion |
| `aexpand(ranked_chunks)` | Async expansion |
| `_fetch_neighbors(metadata)` | SQL: chunks with `chunk_index` between low and high |
| `_finalize_expansion` | Apply token budget via `trim_chunks_to_budget` |

**Example:**

```
Rank-1 chunk is paragraph 5 of a section.
neighbor_chunks=1 → also fetch paragraphs 4 and 6.
expanded_text = paragraph4 + paragraph5 + paragraph6
```

---

## Step 7: `PromptBuilder`

**Job:** Build the messages sent to Groq.

| Method | What it does |
|--------|--------------|
| `build(processed_query, context)` | Calls `fit_prompt_to_budget`, returns `BuiltPrompt` |

**System prompt** (`_SYSTEM_PROMPT`) tells the LLM:

- Answer only from context
- Return JSON: `{"answer": "...", "citations": [...]}`
- Say "I could not find an answer..." if context lacks info

**User prompt shape:**

```
CONTEXT:
--- Source: sample.pdf | Page: 3 | Section: Intro ---
[expanded chunk text]
-------

Question: What is semantic chunking?
```

---

## Step 8: `LLMClient`

**Job:** Call Groq with retries and circuit breaker on repeated failures.

| Method | What it does |
|--------|--------------|
| `generate(prompt)` | Sync wrapper around `asyncio.run(agenerate)` |
| `agenerate(prompt)` | Async call → `LLMResponse` |
| `_agenerate_with_retry` | Retries on HTTP 429/503 |
| `aclose()` | Close HTTP client |

---

## Step 9: `ResponseProcessor`

**Job:** Parse JSON from the LLM and validate citations.

| Method | What it does |
|--------|--------------|
| `process(llm_response, context)` | Returns `ProcessedResponse` |
| `_parse_llm_json(raw)` | Parse JSON; try to repair if wrapped in extra text |
| `_parse_citations(raw, context)` | Flag citations to sources not in retrieved context |

**Example LLM output:**

```json
{
  "answer": "Semantic chunking splits text by meaning rather than fixed size.",
  "citations": [{"source": "sample.pdf", "page": 3}]
}
```

If the LLM cites `other.pdf` but that file was never retrieved → `is_hallucination_risk=True`.

---

## Cache: `QueryCache`

**Job:** Store full `PipelineResult` in Redis (optional).

| Method | What it does |
|--------|--------------|
| `connect()` / `aclose()` | Async Redis connection |
| `get` / `aget` | Read cached result by query hash |
| `set` / `aset` | Save result with TTL (default 1 hour) |
| `invalidate` / `aflush_pattern` | Clear cache after re-ingestion |

Cache key = hash of cleaned query + metadata filters.

---

## Orchestrator: `RAGPipeline`

**Job:** Wire all steps together. This is what you call from your app.

| Method | What it does |
|--------|--------------|
| `__init__(config)` | Sync setup: Postgres, models, BM25 load |
| `create(config)` | **Async factory** for API server |
| `run(raw_query, ...)` | Full pipeline (sync) |
| `arun(raw_query, ...)` | Full pipeline (async) |
| `close()` / `aclose()` | Release DB and HTTP resources |

### `run()` / `arun()` parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `raw_query` | required | User's question string |
| `metadata_filters` | `None` | Optional `{"source_file": "x.pdf"}` |
| `rerank` | `True` | Use cross-encoder reranking |
| `use_cache` | `True` | Read/write Redis cache |

### What you get: `PipelineResult`

```python
result = pipeline.run("What is semantic chunking?")

result.answer              # str — human-readable answer
result.citations           # list[Citation]
result.has_answer            # False if "could not find..."
result.is_hallucination_risk # True if citations look suspicious
result.chunks_used           # ranked chunks that drove the answer
result.llm_context_chunks    # text actually sent to Groq
result.latency_breakdown     # {"embed_ms": 12.3, "retrieve_ms": 45.0, ...}
result.cached                # True if from Redis
result.query_type            # QueryType enum
```

---

## Step 11: `PipelineEvaluator`

**Job:** Measure retrieval quality on test questions (not for normal users).

| Method | What it does |
|--------|--------------|
| `evaluate(cases, k)` | Sync eval: recall@k, precision@k, MRR, NDCG |
| `async_evaluate(...)` | Async version + optional faithfulness check |
| `_ndcg(...)` | Normalized discounted cumulative gain |
| `_check_faithfulness(...)` | Ask LLM if answer only uses context |

**Metrics in plain English:**

- **Recall@K** — Of the chunks you *needed*, how many did we find in top K?
- **Precision@K** — Of top K results, how many were actually relevant?
- **MRR** — How high was the *first* correct chunk? (1st place = 1.0)

---

## FastAPI server (end of `pipeline.py`)

Only loads if FastAPI is installed. App object: `retrieval.pipeline.app`.

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/query` | POST | Run full pipeline; body: `{"query": "...", "metadata_filters": {}, "rerank": true}` |
| `/health` | GET | Postgres, Redis, BM25, embedding status |
| `/stats` | GET | Query count, cache hit rate, avg latency |
| `/cache/flush` | POST | Clear Redis query cache |

### Request/response models

| Model | Fields |
|-------|--------|
| `QueryRequest` | `query`, optional `metadata_filters`, `rerank` |
| `QueryResponse` | `answer`, `citations`, `chunks_used`, `latency_breakdown`, etc. |
| `ChunkResponse` | One ranked chunk in the API response |
| `LLMContextChunkResponse` | Expanded text sent to LLM |

All endpoints require `X-API-Key` header (see `auth.py`).

---

## Environment variables (retrieval-specific)

Set in `.env` or override with `RAG_*` prefix:

| Variable | What it changes |
|----------|-----------------|
| `GROQ_API_KEY` | Required for answer generation |
| `GROQ_MODEL` | Which Groq model |
| `POSTGRES_*` | Database for chunks and vectors |
| `RAG_FINAL_TOP_K` | How many chunks after rerank |
| `RAG_N_DENSE` / `RAG_N_SPARSE` | Search breadth |
| `RAG_RERANK_ENABLED` | `true` / `false` |
| `RAG_MAX_CONTEXT_TOKENS` | Context size cap |
| `RAG_REDIS_HOST` | Redis for caching (optional) |

See parent [README](../README.md) for full project setup.

---

## End-to-end walkthrough (one question)

**User asks:** `"What is semantic chunking?"`

1. **Preprocessor** — valid `QUESTION`, cleaned text unchanged
2. **Embedder** — 384-dim (or similar) vector computed
3. **HybridRetriever** — pgvector finds 20 semantic neighbors; BM25 finds 20 keyword hits; RRF merges to 30
4. **MetadataFilter** — no filters → all 30 kept
5. **Reranker** — cross-encoder picks best 5
6. **ContextBuilder** — each of 5 gets ±1 neighbor chunk; trim if over token budget
7. **PromptBuilder** — system + user prompt with `CONTEXT:` blocks
8. **LLMClient** — Groq returns JSON answer
9. **ResponseProcessor** — parse answer, validate `sample.pdf` citation
10. **Return** — `PipelineResult` with answer, citations, timings

**User asks:** `"hi"`

1. **Preprocessor** — `GREETING`, `is_valid=False`
2. Pipeline stops — returns friendly message, **no database or LLM call**

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|--------------|-----|
| `PostgreSQL has no chunks yet` | No ingestion | Run `python -m semantic_chunker documents/sample.pdf` |
| `GROQ_API_KEY is not set` | Missing env | Add key to `.env` |
| `I could not find an answer...` | No relevant chunks | Check PDF was ingested; try broader question |
| `LLM is temporarily unavailable` | Groq rate limit / circuit breaker | Wait and retry |
| Slow first query | Models loading (embedding, reranker, BM25 corpus) | Normal; later queries faster |
| Cache returns stale answers | Old Redis entries | `POST /cache/flush` or re-ingest |

---

## Where to read the code

| If you want to understand… | Open… |
|----------------------------|--------|
| Full step order | `RAGPipeline.run()` in `pipeline.py` |
| Search logic | `HybridRetriever` in `pipeline.py` |
| Token trimming | `token_budget.py` |
| Groq HTTP | `groq_async.py` |
| CLI | `__main__.py` |
| Web API | bottom of `pipeline.py` (`app`, `/query`) |

---

## Related docs

- [RagPipelines README](../README.md) — project setup, ingestion, env vars
- [Groq API](https://console.groq.com/docs) — LLM provider
- [pgvector](https://github.com/pgvector/pgvector) — vector search in PostgreSQL
