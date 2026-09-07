"""FastAPI service: search UI + JSON/CSV API."""
import csv
import io

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, PlainTextResponse, Response

import config
import search

app = FastAPI(title="Defect Search POC")
_index: search.Index | None = None


def index() -> search.Index:
    global _index
    if _index is None:
        _index = search.Index()
    return _index


def _filters(source, shop, auditor, attribution, date_from, date_to) -> dict:
    return {
        "Source": source, "Shop": shop, "Auditor": auditor,
        "Attribution": attribution, "date_from": date_from, "date_to": date_to,
    }


@app.get("/", include_in_schema=False)
def home():
    return FileResponse("static/index.html")


@app.get("/api/facets")
def facets():
    return index().facets()


@app.get("/api/search")
def api_search(
    q: str = Query(""),
    source: str = "", shop: str = "", auditor: str = "", attribution: str = "",
    date_from: str = "", date_to: str = "",
    limit: int = config.TOP_ROWS,
):
    hits = index().search(q, _filters(source, shop, auditor, attribution, date_from, date_to), limit)
    return {"query": q, "count": len(hits), "results": hits}


@app.get("/api/export.csv")
def export_csv(
    q: str = Query(""),
    source: str = "", shop: str = "", auditor: str = "", attribution: str = "",
    date_from: str = "", date_to: str = "",
    limit: int = 1000,
):
    hits = index().search(q, _filters(source, shop, auditor, attribution, date_from, date_to), limit)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(config.COLUMNS + ["Matched Snippet", "Score"])
    for h in hits:
        w.writerow([h[c] for c in config.COLUMNS] + [h["_matched_snippet"], h["_score"]])
    return Response(
        buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="defect_search.csv"'},
    )


@app.post("/api/reload", response_class=PlainTextResponse)
def reload_index():
    """Pick up a fresh index after the nightly ingest without restarting."""
    global _index
    _index = None
    return f"reloaded: {index().facets()['total_rows']} rows"
