import json, pandas as pd

from finpilot.config import DATA_DIR, CHUNKS_PATH

def chunk_text(text, size=200, overlap=50):
    words = text.split()
    chunks = []
    step = size - overlap
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks

df = pd.read_csv(DATA_DIR / "news.csv")
with open(CHUNKS_PATH, "w") as f:
    for i, row in df.iterrows():
        for j, c in enumerate(chunk_text(row["text"])):
            rec = {"chunk_id": f"{i}_{j}", "ticker": row["ticker"],
                   "date": row["date"], "title": row["title"], "text": c,
                   "index_text": f"[{str(row['date'])[:10]} | {row['ticker']}] {row['title']}\n{c}"}
            f.write(json.dumps(rec) + "\n")