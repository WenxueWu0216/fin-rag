import json
import bm25_search, dense_search, hybrid_search, rerank_search

METHODS = {
    "bm25": bm25_search.search,
    "dense": dense_search.search,
    "hybrid": hybrid_search.search,
    "rerank": rerank_search.search,
}

rows = [json.loads(l) for l in open("eval/rag_questions.jsonl")]
rows = [r for r in rows if r["answerable"]]

for name, search in METHODS.items():
    at5 = at10 = 0
    for r in rows:
        ids = [x[0] for x in search(r["question"], k=10)]
        at5 += r["gold"] in ids[:5]
        at10 += r["gold"] in ids
    print(f"{name:7s} Recall@5: {at5 / len(rows):.0%}   Recall@10: {at10 / len(rows):.0%}")