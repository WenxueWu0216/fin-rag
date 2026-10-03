from sentence_transformers import CrossEncoder
import hybrid_search
from bm25_search import chunks

RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker = CrossEncoder(RERANKER)
id2chunk = {c["chunk_id"]: c for c in chunks}

def search(query, k=5, candidates=30):
    cands = hybrid_search.search(query, k=candidates)
    texts = [id2chunk[r[0]]["text"] for r in cands]
    scores = reranker.predict([(query, t) for t in texts])
    order = scores.argsort()[::-1][:k]
    return [(cands[i][0], round(float(scores[i]), 3), cands[i][2], cands[i][3]) for i in order]

if __name__ == "__main__":
    for r in search("stocks fell sharply before the holiday weekend"):
        print(r)