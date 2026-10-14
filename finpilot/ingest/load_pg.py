"""Load chunks and their precomputed embeddings into Postgres (pgvector).

Idempotent: re-running upserts rows by chunk_id instead of duplicating them.
Run from the repo root: python -m finpilot.ingest.load_pg
"""

import json
from datetime import date

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from finpilot.config import CHUNKS_PATH, DATABASE_URL, EMB_PATH, EMBED_DIM

BATCH_SIZE = 500

UPSERT_SQL = """
INSERT INTO chunks
    (chunk_id, doc_id, source, ticker, published, title, section, text, index_text, embedding)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (chunk_id) DO UPDATE SET
    doc_id     = EXCLUDED.doc_id,
    source     = EXCLUDED.source,
    ticker     = EXCLUDED.ticker,
    published  = EXCLUDED.published,
    title      = EXCLUDED.title,
    section    = EXCLUDED.section,
    text       = EXCLUDED.text,
    index_text = EXCLUDED.index_text,
    embedding  = EXCLUDED.embedding
"""


def parse_date(raw):
    """FNSPID dates look like '2013-10-08 23:00:00 UTC'; keep only the date part."""
    if not raw:
        return None
    return date.fromisoformat(str(raw)[:10])


def to_row(chunk, vec):
    return (
        chunk["chunk_id"],
        chunk["chunk_id"].split("_")[0],  # "905_0" -> article "905"
        "news",
        chunk["ticker"],
        parse_date(chunk.get("date")),
        chunk["title"],
        None,  # section is only used for SEC filings
        chunk["text"],
        chunk["index_text"],
        vec,
    )


def main():
    chunks = [json.loads(line) for line in open(CHUNKS_PATH)]
    emb = np.load(EMB_PATH)

    # Row i of embeddings.npy must belong to line i of chunks.jsonl
    if len(chunks) != len(emb):
        raise ValueError(f"{len(chunks)} chunks but {len(emb)} embeddings; rebuild embeddings")
    if emb.shape[1] != EMBED_DIM:
        raise ValueError(f"embedding dim {emb.shape[1]} != EMBED_DIM {EMBED_DIM}")

    # The connection context manager commits on success and rolls back on error
    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)  # lets psycopg send numpy arrays as pgvector values
        with conn.cursor() as cur:
            for start in range(0, len(chunks), BATCH_SIZE):
                end = start + BATCH_SIZE
                rows = [to_row(c, v) for c, v in zip(chunks[start:end], emb[start:end])]
                cur.executemany(UPSERT_SQL, rows)
                print(f"upserted {min(end, len(chunks))}/{len(chunks)}")
            cur.execute("SELECT COUNT(*) FROM chunks")
            print("rows in table:", cur.fetchone()[0])


if __name__ == "__main__":
    main()
