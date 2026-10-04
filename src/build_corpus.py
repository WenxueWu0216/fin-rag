import json, pandas as pd

def chunk_text(text, size=200, overlap=50):
    words = text.split()
    chunks = []
    step = size - overlap
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks

df = pd.read_csv("data/news.csv")
with open("data/chunks.jsonl", "w") as f:
    for i, row in df.iterrows():
        for j, c in enumerate(chunk_text(row["text"])):
            rec = {"chunk_id": f"{i}_{j}", "ticker": row["ticker"],
                   "date": row["date"], "title": row["title"], "text": c,
                   "index_text": f"[{str(row['date'])[:10]} | {row['ticker']}] {row['title']}\n{c}"}
            f.write(json.dumps(rec) + "\n")