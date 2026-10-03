import pandas as pd

chunks = []
try:
    for chunk in pd.read_csv("data/fnspid_part.csv", chunksize=5000):
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
df.to_csv("data/news.csv", index=False)
print(df.shape)
print(df.head(3))