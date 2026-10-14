import pandas as pd

from finpilot.config import DATA_DIR

chunks = []
try:
    for chunk in pd.read_csv(DATA_DIR / "fnspid_part.csv", chunksize=5000):
        chunks.append(chunk)
except pd.errors.ParserError:
    pass

df = pd.concat(chunks)
df = df[["Date", "Stock_symbol", "Article_title", "Article"]].dropna()
df.columns = ["date", "ticker", "title", "text"]
df = df[df["text"].str.split().str.len() > 100]
before = len(df)
df = df.drop_duplicates(subset="text")
print(f"dedup: {before} -> {len(df)} articles")
df = df.head(2000)
df.to_csv(DATA_DIR / "news.csv", index=False)
print(df.shape)
print(df.head(3))