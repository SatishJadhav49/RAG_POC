"""Shared pieces: text normalization, chunking, embedding (cached), DB schema."""
import hashlib
import re
import sqlite3

import numpy as np

import config

# --------------------------------------------------------------------------
# Text handling
# --------------------------------------------------------------------------

def normalize(text) -> str:
    """Lowercase, strip punctuation, tame repeated characters.

    The repeat rule targets romanized-Hindi noise ("nahiii" -> "nahii"); the
    remaining single/double variance ("awaaz" vs "awaz") is left to the trigram
    index, which matches it naturally.
    """
    t = str(text or "").lower()
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    return re.sub(r"\s+", " ", t).strip()


def squeeze(text) -> str:
    """Collapse every repeated character run to a single character.

    Romanized Hindi varies mostly by doubled vowels - awaaz/awaz/aawaz,
    kharaab/kharab, nahi/nahii. Squeezing both the index and the query maps
    those variants onto one form. Applied only to the trigram index, and
    symmetrically, so English collisions ("all" -> "al") stay self-consistent.
    """
    return re.sub(r"(.)\1+", r"\1", normalize(text))


# Generic words that appear across a defect corpus regardless of the actual
# defect. Kept out of the LEXICAL retrievers only: "AC not working" would
# otherwise rank "Horn not working" top, because BM25 sees "working" as the
# rarer - therefore more informative - term. The semantic retriever still
# receives the untouched query, where the full phrasing does carry meaning.
STOPWORDS = set("""
a an the is are was were be been being not no and or of in on at to for from with by
it its this that there here has have had do does did doing done
working work works issue issues problem problems complaint complaints defect
customer client reported report observed noticed found states said stated
during while after before also very please kindly again still same due getting
hai hain ho hota hoti raha rahi rahe rha rhi me mein se ka ki ke ko par pe
bhi nahi na kar karo kiya gaya aa aata aati jab tab ye yeh wo woh
""".split())


def content_tokens(query: str, learned: set | None = None) -> list[str]:
    """Query tokens with generic terms removed.

    Falls back to the full token list when everything would be dropped, so a
    query like "not working" still returns something.
    """
    toks = normalize(query).split()
    stop = STOPWORDS | (learned or set())
    kept = [t for t in toks if t not in stop]
    return kept or toks


SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")


def chunk(text, min_chars: int = 15, max_chars: int = 500) -> list[str]:
    """Split a paragraph into one chunk per defect mentioned.

    A single description often lists several unrelated defects ("noise from
    front wheel. AC not cooling. paint peeling."). Embedding the whole
    paragraph averages them into mush and the row stops being retrievable by
    any one of them, so each sentence is embedded separately and scores are
    max-pooled back onto the row. Fragments shorter than min_chars carry too
    little signal to stand alone, so they merge into their neighbour.
    """
    parts = [p.strip() for p in SENTENCE_SPLIT.split(str(text or "")) if p.strip()]
    chunks, cur = [], ""
    for p in parts:
        cur = f"{cur} {p}".strip() if cur else p
        if len(cur) >= min_chars:
            chunks.append(cur)
            cur = ""
    if cur:
        if chunks:
            chunks[-1] = f"{chunks[-1]} {cur}"
        else:
            chunks.append(cur)

    out = []
    for c in chunks:  # hard-split anything still oversized
        while len(c) > max_chars:
            cut = c.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            out.append(c[:cut].strip())
            c = c[cut:].strip()
        if c:
            out.append(c)
    return out


def fts_expr(query: str, min_token_len: int = 1, learned: set | None = None) -> str | None:
    """Build a safe FTS5 MATCH expression from the query's content words."""
    toks = [t for t in content_tokens(query, learned) if len(t) >= min_token_len]
    if not toks:
        return None
    return " OR ".join(f'"{t}"' for t in toks)


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS rows (
    row_id              INTEGER PRIMARY KEY,
    source              TEXT,
    problem_description TEXT,
    attribution         TEXT,
    shop                TEXT,
    auditor             TEXT,
    reported_date       TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id INTEGER PRIMARY KEY,
    row_id   INTEGER NOT NULL,
    text     TEXT NOT NULL,
    norm     TEXT NOT NULL,
    vec_pos  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_chunks_row ON chunks(row_id);

-- sha256(normalized chunk) -> vector. Survives rebuilds, so nightly runs only
-- pay to embed genuinely new text.
CREATE TABLE IF NOT EXISTS emb_cache (
    hash TEXT PRIMARY KEY,
    dim  INTEGER NOT NULL,
    vec  BLOB NOT NULL
);

-- LLM relevance verdicts, cached so a query returns a reproducible count.
CREATE TABLE IF NOT EXISTS rerank_cache (
    qhash    TEXT NOT NULL,
    chunk_id INTEGER NOT NULL,
    relevant INTEGER NOT NULL,
    PRIMARY KEY (qhash, chunk_id)
);

-- Terms so common in this corpus that they carry no signal (learned at ingest).
CREATE TABLE IF NOT EXISTS stopwords (term TEXT PRIMARY KEY);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
    USING fts5(norm, content='', tokenize='unicode61');
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_tri
    USING fts5(norm, content='', tokenize='trigram');
"""


def connect(path: str = config.DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------

def _offline_vector(text: str, dim: int) -> np.ndarray:
    """Deterministic fake embedding so the pipeline can run without Azure.

    Not semantic - for smoke-testing ingest/search plumbing only.
    """
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


class Embedder:
    """Embeds text via Azure OpenAI, backed by a persistent content-hash cache."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.dim = config.EMBED_DIM
        self._client = None
        self.api_calls = 0
        self.cache_hits = 0

    @property
    def client(self):
        if self._client is None:
            from openai import AzureOpenAI
            self._client = AzureOpenAI(
                azure_endpoint=config.AZURE_ENDPOINT,
                api_key=config.AZURE_API_KEY,
                api_version=config.AZURE_API_VERSION,
            )
        return self._client

    def _key(self, text: str) -> str:
        """Cache key namespaced by model identity.

        Offline fake vectors and real Azure vectors must never share a key, or
        a single OFFLINE_EMBED=1 run would silently poison the cache and every
        later search would rank against nonsense with no visible error.
        """
        tag = "offline" if config.OFFLINE_EMBED else config.EMBED_DEPLOYMENT
        return hashlib.sha256(f"{tag}|{self.dim}|{text}".encode()).hexdigest()

    def _fetch(self, texts: list[str]) -> list[np.ndarray]:
        """Call the embedding API in batches."""
        out = []
        for i in range(0, len(texts), 64):
            batch = texts[i:i + 64]
            if config.OFFLINE_EMBED:
                out.extend(_offline_vector(t, self.dim) for t in batch)
                continue
            resp = self.client.embeddings.create(
                model=config.EMBED_DEPLOYMENT, input=batch, dimensions=self.dim
            )
            self.api_calls += 1
            out.extend(np.asarray(d.embedding, dtype=np.float32) for d in resp.data)
        return out

    def embed(self, texts: list[str], use_cache: bool = True) -> np.ndarray:
        """Return L2-normalized vectors (so a dot product is cosine similarity)."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        hashes = [self._key(t) for t in texts]
        found: dict[str, np.ndarray] = {}
        if use_cache:
            for i in range(0, len(hashes), 500):
                window = hashes[i:i + 500]
                placeholders = ",".join("?" * len(window))
                q = f"SELECT hash, vec FROM emb_cache WHERE dim=? AND hash IN ({placeholders})"
                for row in self.conn.execute(q, [self.dim, *window]):
                    found[row["hash"]] = np.frombuffer(row["vec"], dtype=np.float32)
            self.cache_hits += len(found)

        missing = {h: t for h, t in zip(hashes, texts) if h not in found}
        if missing:
            miss_h, miss_t = list(missing.keys()), list(missing.values())
            for h, v in zip(miss_h, self._fetch(miss_t)):
                found[h] = v
            if use_cache:
                self.conn.executemany(
                    "INSERT OR REPLACE INTO emb_cache(hash, dim, vec) VALUES (?,?,?)",
                    [(h, self.dim, found[h].tobytes()) for h in miss_h],
                )
                self.conn.commit()

        mat = np.vstack([found[h] for h in hashes])
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return (mat / np.maximum(norms, 1e-9)).astype(np.float32)
