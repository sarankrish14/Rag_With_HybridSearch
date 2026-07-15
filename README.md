# RagPipelines

Standalone RAG project with two pipelines:

- **semantic_chunker** — PDF ingestion (extract → semantic chunk → embed → PostgreSQL)
- **retrieval** — hybrid search, rerank, and Groq answer generation

Runs independently from the parent `saran_rag` repo. All config, database, and model code lives inside this folder.

## Project layout

```
RagPipelines/
├── config.py              # settings and env loading
├── chromastore.py         # optional ChromaDB mirror
├── db/                    # PostgreSQL + pgvector store
├── models/                # embedding model singleton
├── semantic_chunker/      # ingestion pipeline
├── retrieval/             # retrieval pipeline
├── documents/             # sample PDFs for testing
├── chroma_db/             # created at runtime
├── requirements.txt
├── .env.example
└── pyproject.toml
```

## Prerequisites

- Python 3.10+
- PostgreSQL with [pgvector](https://github.com/pgvector/pgvector) extension
- Groq API key (for answer generation in retrieval)

## Setup

```powershell
cd RagPipelines

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
# or: pip install -e .

copy .env.example .env
# Edit .env with your Postgres and Groq credentials

python -c "import nltk; nltk.download('punkt_tab')"
```

Create the database and enable pgvector:

```sql
CREATE DATABASE rag_db;
\c rag_db
CREATE EXTENSION vector;
```

## Ingestion

Process a PDF and store chunks in PostgreSQL:

```powershell
cd RagPipelines
python -m semantic_chunker documents/sample.pdf

# Re-process a duplicate file
python -m semantic_chunker documents/sample.pdf --force

# Run unit tests
python -m semantic_chunker --test
```

Or use the installed script after `pip install -e .`:

```powershell
rag-ingest documents/sample.pdf
```

## Retrieval

Query ingested documents (CLI demo):

```powershell
cd RagPipelines
python -m retrieval --demo

python -m retrieval "What is semantic chunking?"
```

Start the FastAPI server:

```powershell
python -m retrieval --serve
# → http://localhost:8001/docs
```

Or:

```powershell
rag-retrieve --serve
```

## Environment variables

| Variable | Purpose |
|----------|---------|
| `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD` | PostgreSQL connection |
| `GROQ_API_KEY` | LLM answer generation |
| `GROQ_MODEL` | Groq model name (default: `llama-3.3-70b-versatile`) |

See `.env.example` for a full template.

## Python API

```python
from semantic_chunker import semantic_chunk_pdf
from retrieval import RAGPipeline, load_config

chunks = semantic_chunk_pdf("documents/sample.pdf")

pipeline = RAGPipeline(load_config())
result = pipeline.run("What is this document about?")
print(result.answer)
```
