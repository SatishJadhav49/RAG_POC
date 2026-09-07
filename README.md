# Defect Search POC

Hybrid semantic + keyword search over defect rows. Type a defect in plain language
("tyre damage", "brake awaaz", "AC not cooling") and get back the full source rows -
Source, Problem Description, Attribution, Shop, Auditor, Reported Date.

## How it works

Excel -> per-sentence chunks -> embeddings (Azure `text-embedding-3-large`) -> SQLite + numpy index.

A query runs through three retrievers, fused with Reciprocal Rank Fusion:

| Retriever  | Covers | Example |
|---|---|---|
| `semantic` | meaning / synonyms | "tyre damage" finds "rim bent", "tread worn unevenly" |
| `keyword`  | exact terms, part numbers, codes | "P0420", "sidewall" |
| `spelling` | romanized-Hindi variants | "awaz" finds "awaaz", "aawaz" |

Three design points worth knowing:

- **Generic words are stripped from the lexical retrievers.** Searching "AC not
  working" would otherwise rank "Horn not working" first: BM25 sees `working` as
  rarer than `ac`, therefore more informative, and has no idea `ac` is the subject.
  A static `STOPWORDS` list plus terms learned at ingest (anything appearing in
  more than `STOP_DF` of chunks) keeps filler out of keyword and spelling search.
  The semantic retriever still gets the untouched query, where phrasing does matter.
  If every token is filler ("not working"), the filter falls back to the full query.

- **Per-sentence chunking.** One description often lists several unrelated defects.
  Embedding the whole paragraph averages them into mush and the row stops being
  findable by any one of them. Each sentence is embedded separately; scores are
  max-pooled back onto the row, so a row is retrievable by *any* defect it mentions
  and still appears exactly once. The UI shows which sentence matched.
- **A similarity floor (`MIN_SIM`).** Vector search is top-k with no natural cutoff,
  so a nonsense query would otherwise still return 30 confident-looking rows.

## Optional: LLM relevance filter

Set `RERANK=1` to add a gpt-4o pass between retrieval and display. It reads the
matched sentence of each candidate and decides which genuinely describe the
queried defect - catching what no lexical rule can, e.g. that "Transit damage
suspected" is not a tyre defect while "Alloy wheel damaged" is.

Three properties worth understanding before you rely on it:

- **It can only raise precision, never recall.** The filter judges what retrieval
  already found and can never add a row back. So enabling it *widens* the
  candidate pool to `RERANK_CANDIDATES` (100) and filters down, rather than
  filtering the top 30. Recall has to be won before this step runs.
- **Verdicts are cached and the model runs at `temperature=0`,** so a query
  returns a reproducible count. If a number from this tool lands in an audit
  report, it must not change between runs.
- **Excluded rows are collapsed, not deleted.** The UI shows them behind a
  "show N excluded" toggle so nothing silently disappears from an audit search.
  CSV export contains only the kept rows.

Cost is roughly a few hundred tokens per search; negligible at 10 users. It adds
about 1-3 seconds of latency, so it is a per-search toggle in the UI (`?rerank=true|false`)
and you can A/B it against plain hybrid search.

If gpt-4o is unreachable the search itself still works - turn the toggle off.

## Setup

```bash
pip install -r requirements.txt
copy .env.example .env      # then fill in your Azure values
```

## Run

```bash
python make_sample.py      # only for demo data - skip once you have the real file
python ingest.py           # build the index
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000

To try the plumbing before Azure credentials are ready, set `OFFLINE_EMBED=1`.
That substitutes deterministic fake vectors: ingest and search run end to end,
but the `semantic` retriever returns nothing useful. Keyword and spelling still work.

## Using your real data

Drop the export at `data/defects.xlsx` with these exact column headers and re-run `python ingest.py`:

```
Source | Problem Description | Attribution | Shop | Auditor | Reported Date
```

Moving to SQL later means replacing `load_dataframe()` in `ingest.py` with a
`pd.read_sql(...)` call against your stored procedure. Nothing else changes.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/search?q=...` | JSON rows; optional `source`, `shop`, `auditor`, `attribution`, `date_from`, `date_to`, `limit`, `rerank` |
| `GET /api/export.csv?q=...` | Same results as CSV |
| `GET /api/facets` | Filter dropdown values |
| `POST /api/reload` | Reload the index after a nightly ingest, no restart |

## Nightly refresh (Windows Server)

Task Scheduler, daily:

```bat
cd /d C:\Satish\RAG-POC && python ingest.py && curl -X POST http://localhost:8000/api/reload
```

`ingest.py` rebuilds rows and indexes but keeps the embedding cache, so only genuinely
new text costs an API call. On the sample data 245 chunks collapse to 93 distinct
embeddings - real dealer verbatims repeat far more heavily than that.

## Tuning knobs

All in `config.py` / `.env`:

- `MIN_SIM` (0.30) - cosine floor. Raise if results feel loose, lower if too strict.
  **Tune this first once real embeddings are in.**
- `STOP_DF` (0.35) - a term in more than this share of chunks is treated as filler.
  Lower it if generic words still leak into results on the real corpus.
- `TOP_ROWS` (30) - rows returned
- `CAND_PER_LIST` (200) - candidates per retriever before fusion
- `EMBED_DIM` (1024) - Matryoshka truncation of `text-embedding-3-large`. At the
  full 3072 you would need ~245 GB of vectors at 5M rows; 1024 keeps it near 82 GB
  with little quality loss.

## Scaling notes

Vector search is exact brute force (`vectors @ query`) held in RAM - roughly 20 ms at
50K rows, and no index-recall loss. That holds to about 1-2M rows on a 64 GB box.
Past that, swap `Index._vector` in `search.py` for FAISS or hnswlib; nothing else
in the codebase needs to know. Deliberately not done now - at this size an ANN index
would be slower and less accurate than the brute-force scan.

## Known limits (POC)

- No auth. Add it before anyone outside the team gets the URL.
- The index reloads into memory on `/api/reload`; with 10 users a single uvicorn
  process is fine, and no worker pool is needed.
- `squeeze()` collapses repeated letters for the spelling index, which also merges
  some English words ("all" -> "al"). Applied symmetrically to index and query, so
  it is self-consistent, but it does slightly loosen keyword matching.
