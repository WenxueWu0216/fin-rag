"""Measure how closely approximate (HNSW) search matches exact (numpy) search.

For each eval question, compute the overlap between the top-k chunk IDs from
exact brute-force search and from pgvector. 100% means HNSW lost nothing.
For the worst question, print what was swapped so you can tell a real miss
from a near-tie at the rank-k boundary.
Run from the repo root: python -m finpilot.eval.ann_check
"""

import json
import time

from finpilot.config import EVAL_DIR
from finpilot.retrieval import dense, dense_pg

K = 10

questions = [json.loads(line)["question"] for line in open(EVAL_DIR / "rag_questions.jsonl")]

overlaps = []
worst = None
hnsw_seconds = 0.0
for q in questions:
    # Fetch extra exact results so we can look up scores just below the cutoff
    exact_ranked = dense.search(q, k=K + 20)
    exact_scores = {cid: score for cid, score, _, _ in exact_ranked}
    exact = {r[0] for r in exact_ranked[:K]}
    t0 = time.perf_counter()
    approx = {r[0] for r in dense_pg.search(q, k=K)}
    hnsw_seconds += time.perf_counter() - t0

    overlap = len(exact & approx) / K
    overlaps.append(overlap)
    if worst is None or overlap < worst["overlap"]:
        worst = {
            "question": q,
            "overlap": overlap,
            "cutoff": exact_ranked[K - 1][1],  # exact score of the k-th result
            "missed": sorted(exact - approx),
            "extra": sorted(approx - exact),
            "scores": exact_scores,
        }

print(f"questions: {len(overlaps)}, hnsw.ef_search: {dense_pg.EF_SEARCH}")
print(f"ANN recall@{K} vs exact: mean {sum(overlaps) / len(overlaps):.1%}, min {min(overlaps):.0%}")
print(f"HNSW latency (incl. query encoding): {hnsw_seconds / len(overlaps) * 1000:.1f} ms/query")

if worst["overlap"] == 1.0:
    raise SystemExit  # every question matched exactly; nothing to diagnose

print(f"\nworst question: {worst['question']}")
print(f"exact score of rank {K}: {worst['cutoff']}")
for cid in worst["missed"]:
    print(f"  missed by HNSW  {cid}  exact score {worst['scores'][cid]}")
for cid in worst["extra"]:
    print(f"  HNSW instead    {cid}  exact score {worst['scores'].get(cid, 'below top-' + str(K + 20))}")
