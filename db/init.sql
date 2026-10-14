-- Schema for FinPilot. Executed once when the database is first created.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   TEXT PRIMARY KEY,               -- e.g. "905_0"; matches eval gold labels
    doc_id     TEXT NOT NULL,
    source     TEXT NOT NULL DEFAULT 'news',   -- 'news' or 'sec'
    ticker     TEXT NOT NULL,
    published  DATE,
    title      TEXT,
    section    TEXT,                           -- filing section (e.g. 'Item 1A'); NULL for news
    text       TEXT NOT NULL,                  -- raw chunk text passed to the LLM
    index_text TEXT NOT NULL,                  -- contextual header + text, used for retrieval
    embedding  vector(384) NOT NULL            -- dimension must match EMBED_DIM in config.py
);

-- Approximate nearest-neighbor index; queries must use the cosine operator (<=>)
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Metadata filtering by ticker and date
CREATE INDEX IF NOT EXISTS chunks_ticker_date
    ON chunks (ticker, published);
