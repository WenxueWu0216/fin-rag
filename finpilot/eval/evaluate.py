import json, sys, time

from finpilot.config import EVAL_DIR

mode = sys.argv[1] if len(sys.argv) > 1 else "bm25"
if mode == "dense":
    from finpilot.retrieval.dense import search
elif mode == "hybrid":
    from finpilot.retrieval.hybrid import search
elif mode == "rerank":
    from finpilot.retrieval.rerank import search
else:
    from finpilot.retrieval.bm25 import search
print("mode:", mode)
K = 5
queries = [json.loads(l) for l in open(EVAL_DIR / "queries.jsonl")]

hits, rr_sum = 0, 0.0
t0 = time.perf_counter()
for q in queries:
    results = search(q["query"], k=K)
    ids = [r[0] for r in results]
    relevant = set(q["relevant"])
    for rank, cid in enumerate(ids, start=1):
        if cid in relevant:
            hits += 1
            rr_sum += 1 / rank
            break

n = len(queries)
print(f"Hit@{K}: {hits / n:.3f}")
print(f"MRR@{K}: {rr_sum / n:.3f}")
ms = (time.perf_counter() - t0) / n * 1000
print(f"avg latency: {ms:.1f} ms/query")