import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.resolve()
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(BASE_DIR / "chroma_nasa"))
DATA_CSV = os.getenv("DATA_CSV", str(BASE_DIR / "data" / "papers.csv"))

# Text splitting
TEXT_CHUNK_SIZE = int(os.getenv("TEXT_CHUNK_SIZE", "500"))
TEXT_CHUNK_OVERLAP = int(os.getenv("TEXT_CHUNK_OVERLAP", "50"))

# Embeddings
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# LLM
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "7895"))

# Retrival defaults
TOP_K = int(os.getenv("TOP_K", "4"))
POOL_K = int(os.getenv("POOL_K", "12"))
DIST_THRESHOLD = float(os.getenv("DIST_THRESHOLD", "0.30"))  # cosine *distance* (lower = better)

# Misc
DEVICE = "cuda" if os.getenv("FORCE_CPU", "0") != "1" and \
                 (os.getenv("CUDA_VISIBLE_DEVICES") or "").strip() != "" else "cpu"
