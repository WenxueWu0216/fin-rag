"""Central configuration: paths, model names, and database settings."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Repo root: finpilot/config.py -> finpilot/ -> repo root
ROOT = Path(__file__).resolve().parent.parent

# Values in .env take precedence over existing environment variables
load_dotenv(ROOT / ".env", override=True)

# Paths
DATA_DIR = ROOT / "data"
EVAL_DIR = ROOT / "evalsets"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
EMB_PATH = DATA_DIR / "embeddings.npy"

# Models
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384  # must match the vector(...) column in db/init.sql
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "claude-haiku-4-5-20251001"

# Database (port must match the host port in docker-compose.yml; override with DATABASE_URL in .env)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://finpilot:finpilot@localhost:5433/finpilot",
)
