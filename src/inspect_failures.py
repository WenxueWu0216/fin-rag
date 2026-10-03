import json
from bm25_search import search, chunks

queries = [json.loads(l) for l in open("eval/queries.jsonl")]
id2chunk = {c["chunk_id"]: c for c in chunks}

shown = 0
for q in queries:
    results = search(q["query"], k=5)
    if any(r[0] in q["relevant"] for r in results):
        continue
    print("QUERY:", q["query"])
    print("  correct:", id2chunk[q["relevant"][0]]["text"][:150])
    for r in results[:3]:
        print("  got:", r[2], "|", r[3])
    shown += 1
    if shown == 5:
        break