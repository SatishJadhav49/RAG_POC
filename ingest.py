"""Excel -> chunks -> embeddings -> searchable index.

Safe to re-run nightly: rows and FTS tables are rebuilt from scratch, but the
embedding cache persists, so only genuinely new text costs an API call.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

import config
import core


def load_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    missing = [c for c in config.COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"Excel file is missing required column(s): {missing}\nFound: {list(df.columns)}")
    df = df[config.COLUMNS].copy()
    # Keep dates as plain strings so they round-trip cleanly to JSON/CSV.
    df["Reported Date"] = df["Reported Date"].apply(
        lambda v: v.date().isoformat() if hasattr(v, "date") else ("" if pd.isna(v) else str(v))
    )
    for c in config.COLUMNS:
        df[c] = df[c].fillna("").astype(str).str.strip()
    return df[df[config.TEXT_COL] != ""].reset_index(drop=True)


def main() -> None:
    t0 = time.time()
    os.makedirs(config.INDEX_DIR, exist_ok=True)

    df = load_dataframe(config.DATA_FILE)
    print(f"loaded {len(df)} rows from {config.DATA_FILE}")

    conn = core.connect()
    # Contentless FTS5 tables reject plain DELETE; 'delete-all' is the documented reset.
    conn.executescript(
        "DELETE FROM rows;"
        " DELETE FROM chunks;"
        " INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all');"
        " INSERT INTO chunks_tri(chunks_tri) VALUES('delete-all');"
    )

    conn.executemany(
        "INSERT INTO rows(row_id, source, problem_description, attribution, shop, auditor, reported_date)"
        " VALUES (?,?,?,?,?,?,?)",
        [(i, r["Source"], r[config.TEXT_COL], r["Attribution"], r["Shop"], r["Auditor"], r["Reported Date"])
         for i, r in df.iterrows()],
    )

    # Build the chunk list. vec_pos is the row offset into vectors.npy.
    records = []
    for i, r in df.iterrows():
        for text in core.chunk(r[config.TEXT_COL]):
            records.append((len(records), int(i), text, core.normalize(text), len(records)))
    if not records:
        raise SystemExit("no chunks produced - is the Problem Description column populated?")
    print(f"produced {len(records)} chunks ({len(records)/len(df):.1f} per row)")

    conn.executemany("INSERT INTO chunks(chunk_id, row_id, text, norm, vec_pos) VALUES (?,?,?,?,?)", records)
    conn.executemany("INSERT INTO chunks_fts(rowid, norm) VALUES (?,?)", [(r[0], r[3]) for r in records])
    conn.executemany("INSERT INTO chunks_tri(rowid, norm) VALUES (?,?)", [(r[0], core.squeeze(r[3])) for r in records])
    # Learn corpus-specific filler: terms so common here they cannot
    # discriminate between defects. Complements the static STOPWORDS list and
    # adapts automatically to however your dealers actually write.
    from collections import Counter
    df = Counter()
    for r in records:
        df.update(set(r[3].split()))
    learned = sorted(t for t, n in df.items() if n / len(records) > config.STOP_DF)
    conn.execute("DELETE FROM stopwords")
    conn.executemany("INSERT INTO stopwords(term) VALUES (?)", [(t,) for t in learned])
    if learned:
        print(f"learned {len(learned)} corpus stopwords: {', '.join(learned[:12])}"
              + (" ..." if len(learned) > 12 else ""))
    conn.commit()

    mode = "OFFLINE (fake vectors)" if config.OFFLINE_EMBED else f"{config.EMBED_DEPLOYMENT} @ {config.EMBED_DIM}d"
    print(f"embedding via {mode} ...")
    emb = core.Embedder(conn)
    vectors = emb.embed([r[2] for r in records])
    np.save(config.VEC_PATH, vectors)

    conn.commit()
    conn.close()
    size_mb = vectors.nbytes / 1e6
    print(
        f"done in {time.time() - t0:.1f}s | cache hits {emb.cache_hits}/{len(records)} | "
        f"api calls {emb.api_calls} | vectors {vectors.shape} ({size_mb:.1f} MB)"
    )


if __name__ == "__main__":
    sys.exit(main())
