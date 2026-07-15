-- PostgreSQL schema for document registry, chunk metadata, and embeddings.
-- Requires: CREATE EXTENSION vector; (run once per database)

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    doc_id              TEXT PRIMARY KEY,
    file_name           TEXT NOT NULL,
    file_path           TEXT,
    file_hash           TEXT NOT NULL UNIQUE,
    page_count          INTEGER,
    status              TEXT NOT NULL DEFAULT 'complete',
    docling_metadata    JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id            TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index         INTEGER NOT NULL,
    text                TEXT NOT NULL,
    embedding           vector(768) NOT NULL,
    sentence_count      INTEGER,
    char_count          INTEGER,
    page_no             INTEGER,
    source              TEXT,
    threshold_used      DOUBLE PRECISION,
    metadata            JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_index ON chunks(doc_id, chunk_index);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks
    USING hnsw (embedding vector_cosine_ops);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'chunks'
          AND column_name = 'text_tsv'
    ) THEN
        ALTER TABLE chunks ADD COLUMN text_tsv tsvector
            GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_chunks_text_tsv ON chunks USING GIN (text_tsv);
