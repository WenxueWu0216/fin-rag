import json

from finpilot.config import EVAL_DIR
from finpilot.retrieval import bm25, dense, hybrid, rerank

METHODS = {
    "bm25": bm25.search,
    "dense": dense.search,
    "hybrid": hybrid.search,
    "rerank": rerank.search,
}

rows = [json.loads(l) for l in open(EVAL_DIR / "rag_questions.jsonl")]
rows = [r for r in rows if r["answerable"]]

for name, search in METHODS.items():
    at5 = at10 = 0
    for r in rows:
        ids = [x[0] for x in search(r["question"], k=10)]
        at5 += r["gold"] in ids[:5]
        at10 += r["gold"] in ids
    print(f"{name:7s} Recall@5: {at5 / len(rows):.0%}   Recall@10: {at10 / len(rows):.0%}")