"""Hybrid retrieval: vectors + word BM25 + trigram BM25, fused with RRF.

Each retriever covers a different failure mode:
  vectors  - meaning ("tyre damage" finds "rim bent", "tread worn")
  word FTS - exact terms, part numbers, model codes
  trigram  - spelling variance, which romanized Hindi produces constantly
             ("awaaz" / "awaz" / "aawaz")
Chunk scores are max-pooled onto their row, so a row is retrievable by any
single defect it mentions and is still returned exactly once.
"""
import numpy as np

import config
import core
import rerank as rerank_mod

FIELDS = ["Source", "Problem Description", "Attribution", "Shop", "Auditor", "Reported Date"]


class Index:
    def __init__(self):
        self.conn = core.connect()
        self.vectors = np.load(config.VEC_PATH)
        order = self.conn.execute("SELECT chunk_id FROM chunks ORDER BY vec_pos").fetchall()
        self.chunk_ids = np.array([r["chunk_id"] for r in order], dtype=np.int64)
        self.embedder = core.Embedder(self.conn)
        self.learned_stop = {r["term"] for r in self.conn.execute("SELECT term FROM stopwords")}

    # -- individual retrievers -------------------------------------------

    def _vector(self, query: str, k: int) -> list[int]:
        qv = self.embedder.embed([query])[0]
        scores = self.vectors @ qv
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        # Vector search is top-k with no natural cutoff, so an unrelated query
        # would still return k rows. The floor keeps "no good match" honest.
        return [int(self.chunk_ids[i]) for i in top if scores[i] >= config.MIN_SIM]

    def _fts(self, table: str, query: str, k: int, min_token_len: int, squeeze: bool = False) -> list[int]:
        learned = {core.squeeze(t) for t in self.learned_stop} if squeeze else self.learned_stop
        expr = core.fts_expr(core.squeeze(query) if squeeze else query, min_token_len, learned)
        if not expr:
            return []
        sql = f"SELECT rowid FROM {table} WHERE {table} MATCH ? ORDER BY bm25({table}) LIMIT ?"
        try:
            return [r["rowid"] for r in self.conn.execute(sql, (expr, k))]
        except Exception:
            return []

    # -- fusion -----------------------------------------------------------

    def search(self, query: str, filters: dict | None = None, top_rows: int = config.TOP_ROWS,
               rerank: bool | None = None) -> list[dict]:
        query = (query or "").strip()
        if not query:
            return []

        k = config.CAND_PER_LIST
        lists = {
            "semantic": self._vector(query, k),
            "keyword": self._fts("chunks_fts", query, k, min_token_len=1),
            "spelling": self._fts("chunks_tri", query, k, min_token_len=3, squeeze=True),
        }

        # Reciprocal Rank Fusion - rank-based, so the three scales never need tuning.
        fused: dict[int, float] = {}
        matched_by: dict[int, set] = {}
        for name, ids in lists.items():
            for rank, cid in enumerate(ids):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (config.RRF_K + rank + 1)
                matched_by.setdefault(cid, set()).add(name)
        if not fused:
            return []

        # Max-pool chunks onto their parent row.
        placeholders = ",".join("?" * len(fused))
        chunk_rows = self.conn.execute(
            f"SELECT chunk_id, row_id, text FROM chunks WHERE chunk_id IN ({placeholders})",
            list(fused),
        ).fetchall()

        best: dict[int, dict] = {}
        for cr in chunk_rows:
            rid, cid = cr["row_id"], cr["chunk_id"]
            cand = {"score": fused[cid], "snippet": cr["text"], "matched_by": matched_by[cid], "cid": cid}
            if rid not in best or cand["score"] > best[rid]["score"]:
                best[rid] = cand

        ranked = sorted(best.items(), key=lambda kv: -kv[1]["score"])
        use_llm = config.RERANK if rerank is None else rerank
        if not use_llm:
            return self._materialize(ranked, filters or {}, top_rows)

        # Over-fetch, then let the LLM cut. Filtering can only ever remove rows,
        # so the candidate pool has to be wide enough to hold the recall.
        rows = self._materialize(ranked, filters or {}, max(top_rows, config.RERANK_CANDIDATES))
        verdicts = rerank_mod.judge(
            self.conn, query,
            [{"id": r["_cid"], "text": r["_matched_snippet"]} for r in rows],
        )
        for r in rows:
            r["_relevant"] = verdicts.get(r["_cid"], True)
        keep = [r for r in rows if r["_relevant"]][:top_rows]
        drop = [r for r in rows if not r["_relevant"]]
        return keep + drop

    # -- row hydration ----------------------------------------------------

    def _materialize(self, ranked: list, filters: dict, top_rows: int) -> list[dict]:
        """Attach every source column to each hit, applying filters first."""
        row_ids = [rid for rid, _ in ranked]
        placeholders = ",".join("?" * len(row_ids))
        by_id = {
            r["row_id"]: r
            for r in self.conn.execute(f"SELECT * FROM rows WHERE row_id IN ({placeholders})", row_ids)
        }

        col = {"Source": "source", "Shop": "shop", "Auditor": "auditor", "Attribution": "attribution"}
        out = []
        for rid, meta in ranked:
            r = by_id.get(rid)
            if r is None:
                continue
            if any(r[col[f]] != v for f, v in filters.items() if f in col and v):
                continue
            date = r["reported_date"]
            if filters.get("date_from") and date < filters["date_from"]:
                continue
            if filters.get("date_to") and date > filters["date_to"]:
                continue
            out.append({
                "Source": r["source"],
                "Problem Description": r["problem_description"],
                "Attribution": r["attribution"],
                "Shop": r["shop"],
                "Auditor": r["auditor"],
                "Reported Date": date,
                "_score": round(meta["score"], 5),
                "_matched_snippet": meta["snippet"],
                "_matched_by": sorted(meta["matched_by"]),
                "_cid": meta["cid"],
                "_relevant": True,
            })
            if len(out) >= top_rows:
                break
        return out

    def facets(self) -> dict:
        out = {}
        for label, c in [("Source", "source"), ("Shop", "shop"), ("Auditor", "auditor"), ("Attribution", "attribution")]:
            rows = self.conn.execute(f"SELECT DISTINCT {c} AS v FROM rows WHERE v != '' ORDER BY v")
            out[label] = [r["v"] for r in rows]
        out["total_rows"] = self.conn.execute("SELECT COUNT(*) c FROM rows").fetchone()["c"]
        return out
