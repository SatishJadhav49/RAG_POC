import os
from dotenv import load_dotenv

load_dotenv()

DATA_FILE = os.getenv("DATA_FILE", "data/defects.xlsx")
INDEX_DIR = os.getenv("INDEX_DIR", "index")
DB_PATH = os.path.join(INDEX_DIR, "defects.db")
VEC_PATH = os.path.join(INDEX_DIR, "vectors.npy")

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
EMBED_DEPLOYMENT = os.getenv("AZURE_EMBED_DEPLOYMENT", "text-embedding-3-large")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
OFFLINE_EMBED = os.getenv("OFFLINE_EMBED", "0") == "1"

# Exact column names expected in the Excel file.
COLUMNS = ["Source", "Problem Description", "Attribution", "Shop", "Auditor", "Reported Date"]
TEXT_COL = "Problem Description"

TOP_ROWS = 30          # rows returned to the UI
CAND_PER_LIST = 200    # candidates pulled from each retriever before fusion
RRF_K = 60             # reciprocal-rank-fusion constant
MIN_SIM = float(os.getenv("MIN_SIM", "0.30"))  # cosine floor; tune during POC
STOP_DF = float(os.getenv("STOP_DF", "0.35"))   # terms in >35% of chunks are treated as filler

# --- optional LLM relevance filter (Azure gpt-4o) ---
RERANK = os.getenv("RERANK", "0") == "1"
CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "100"))  # widened pool: the
# filter can only remove rows, so recall must be won before it runs
RERANK_BATCH = int(os.getenv("RERANK_BATCH", "50"))
