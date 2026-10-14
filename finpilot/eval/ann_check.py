"""Measure how closely approximate (HNSW) search matches exact (numpy) search.

For each eval question, compute the overlap between the top-k chunk IDs from
exact brute-force search and from pgvector. 100% means HNSW lost nothing.
Run from the repo root: python -m finpilot.eval.ann_check
"""

import json

from finpilot.config import EVAL_DIR
from finpilot.retrieval import dense, dense_pg

K = 10

questions = [json.loads(line)["question"] for line in open(EVAL_DIR / "rag_questions.jsonl")]

overlaps = []
for q in questions:
    exact = {r[0] for r in dense.search(q, k=K)}
    approx = {r[0] for r in dense_pg.search(q, k=K)}
    overlaps.append(len(exact & approx) / K)

print(f"questions: {len(overlaps)}")
print(f"ANN recall@{K} vs exact: mean {sum(overlaps) / len(overlaps):.1%}, min {min(overlaps):.0%}")
