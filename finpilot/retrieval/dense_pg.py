"""Dense retrieval backed by pgvector's HNSW index.

Same interface as retrieval.dense: search(query, k) -> [(chunk_id, score, ticker, title)].
"""

import os

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from finpilot.config import DATABASE_URL, EMBED_MODEL

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# HNSW returns at most ef_search rows, so it must be >= the largest k we ask for
# (hybrid search requests a pool of 50). Higher values trade speed for recall.
# Override with the HNSW_EF_SEARCH env var to experiment.
EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "100"))

model = SentenceTransformer(EMBED_MODEL)

# One connection per process; autocommit avoids holding an open transaction on reads
conn = psycopg.connect(DATABASE_URL, autocommit=True)
register_vector(conn)
conn.execute(f"SET hnsw.ef_search = {EF_SEARCH}")

# <=> is cosine distance; it must match the vector_cosine_ops index for HNSW to be used
SEARCH_SQL = """
SELECT chunk_id, 1 - (embedding <=> %(q)s) AS score, ticker, title
FROM chunks
ORDER BY embedding <=> %(q)s
LIMIT %(k)s
"""


def search(query, k=5):
    if k > EF_SEARCH:
        raise ValueError(f"k={k} exceeds hnsw.ef_search={EF_SEARCH}")
    q = model.encode(QUERY_PREFIX + query, normalize_embeddings=True)
    rows = conn.execute(SEARCH_SQL, {"q": q, "k": k}).fetchall()
    return [(cid, round(float(score), 3), ticker, title) for cid, score, ticker, title in rows]


if __name__ == "__main__":
    for r in search("stocks fell sharply before the holiday weekend"):
        print(r)
