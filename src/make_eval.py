import json, random

random.seed(42)
chunks = [json.loads(l) for l in open("data/chunks.jsonl")]

title_to_ids = {}
for c in chunks:
    title_to_ids.setdefault(c["title"], []).append(c["chunk_id"])

titles = sorted(title_to_ids)
sample = random.sample(titles, min(200, len(titles)))

with open("eval/queries.jsonl", "w") as f:
    for t in sample:
        f.write(json.dumps({"query": t, "relevant": title_to_ids[t]}) + "\n")

print(len(titles), "unique titles; wrote", len(sample), "queries")