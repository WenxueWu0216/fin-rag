import json
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = "BAAI/bge-small-en-v1.5"

chunks = [json.loads(l) for l in open("data/chunks.jsonl")]
texts = [c["index_text"] for c in chunks]

model = SentenceTransformer(MODEL)
emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

np.save("data/embeddings.npy", emb.astype("float32"))
print(emb.shape)