import json
import numpy as np
from sentence_transformers import SentenceTransformer

from finpilot.config import CHUNKS_PATH, EMB_PATH, EMBED_MODEL

chunks = [json.loads(l) for l in open(CHUNKS_PATH)]
texts = [c["index_text"] for c in chunks]

model = SentenceTransformer(EMBED_MODEL)
emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

np.save(EMB_PATH, emb.astype("float32"))
print(emb.shape)