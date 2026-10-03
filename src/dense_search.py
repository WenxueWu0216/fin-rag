import json
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

chunks = [json.loads(l) for l in open("data/chunks.jsonl")]
emb = np.load("data/embeddings.npy")
model = SentenceTransformer(MODEL)

def search(query, k=5):
    q = model.encode(QUERY_PREFIX + query, normalize_embeddings=True)
    scores = emb @ q
    top = np.argsort(-scores)[:k]
    return [(chunks[i]["chunk_id"], round(float(scores[i]), 3), chunks[i]["ticker"], chunks[i]["title"]) for i in top]

if __name__ == "__main__":
    for r in search("stocks fell sharply before the holiday weekend"):
        print(r)
        