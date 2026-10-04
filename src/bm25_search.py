import json
from rank_bm25 import BM25Okapi
import re

chunks = [json.loads(l) for l in open("data/chunks.jsonl")]
def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())
tokenized = [tokenize(c["index_text"]) for c in chunks]
bm25 = BM25Okapi(tokenized)

def search(query, k=5):
    scores = bm25.get_scores(tokenize(query))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(chunks[i]["chunk_id"], round(scores[i], 2), chunks[i]["ticker"], chunks[i]["title"]) for i in top]

if __name__ == "__main__":
    for r in search("Apple earnings beat expectations"):
        print(r)