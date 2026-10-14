from sentence_transformers import CrossEncoder

from finpilot.config import RERANK_MODEL
from finpilot.retrieval import hybrid
from finpilot.retrieval.bm25 import chunks

reranker = CrossEncoder(RERANK_MODEL)
id2chunk = {c["chunk_id"]: c for c in chunks}

def search(query, k=5, candidates=30):
    cands = hybrid.search(query, k=candidates)
    texts = [id2chunk[r[0]]["index_text"] for r in cands]
    scores = reranker.predict([(query, t) for t in texts])
    order = scores.argsort()[::-1][:k]
    return [(cands[i][0], round(float(scores[i]), 3), cands[i][2], cands[i][3]) for i in order]

if __name__ == "__main__":
    for r in search("stocks fell sharply before the holiday weekend"):
        print(r)