import json
import numpy as np
from sentence_transformers import SentenceTransformer

from finpilot.config import CHUNKS_PATH, EMB_PATH, EMBED_MODEL

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

chunks = [json.loads(l) for l in open(CHUNKS_PATH)]
emb = np.load(EMB_PATH)
model = SentenceTransformer(EMBED_MODEL)

def search(query, k=5):
    q = model.encode(QUERY_PREFIX + query, normalize_embeddings=True)
    scores = emb @ q
    top = np.argsort(-scores)[:k]
    return [(chunks[i]["chunk_id"], round(float(scores[i]), 3), chunks[i]["ticker"], chunks[i]["title"]) for i in top]

if __name__ == "__main__":
    for r in search("stocks fell sharply before the holiday weekend"):
        print(r)
        