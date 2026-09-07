"""Optional LLM relevance filter (Azure gpt-4o).

Retrieval is tuned for recall and deliberately over-fetches; this pass decides
which candidates actually describe the queried defect. It only ever *labels*
rows - it never rewrites them, and never adds rows retrieval did not find, so
it can raise precision but can never repair a recall miss. That is why
search.py widens the candidate pool when this is enabled.

Verdicts are cached and the model runs at temperature 0, so a given query
returns a reproducible count - which matters when the number ends up in an
audit report.
"""
import hashlib
import json

import config

SYSTEM = (
    "You screen vehicle defect records for an audit search tool. "
    "Records may be in English, Hindi written in Latin script, or a mix, and contain "
    "dealer spelling mistakes."
)

INSTRUCTIONS = """Decide which records describe the SAME defect concern as the query.

INCLUDE a record if it refers to the same component or failure mode as the query,
including synonyms, related parts, and Hindi/Hinglish phrasing.
  e.g. query "tyre damage" -> include tyre, tire, wheel, rim, alloy, tread, puncture, sidewall
EXCLUDE a record about a different component, even when it shares generic words
like "not working", "noise", "damage", or "problem".
  e.g. query "AC not working" -> EXCLUDE "Horn not working"

Return ONLY JSON: {"relevant": [<ids of matching records>]}"""


def _stub(query: str, candidates: list[dict]) -> dict[int, bool]:
    """Offline stand-in: keeps candidates sharing a content word with the query.

    Exercises the filtering/caching path without calling Azure. Not a
    substitute for the model - it cannot tell "wheel" relates to "tyre".
    """
    import core
    want = set(core.content_tokens(query))
    return {c["id"]: bool(want & set(core.normalize(c["text"]).split())) for c in candidates}


def _ask(client, query: str, batch: list[dict]) -> set[int]:
    listing = "\n".join(f'{c["id"]}. {c["text"]}' for c in batch)
    resp = client.chat.completions.create(
        model=config.CHAT_DEPLOYMENT,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f'Query: "{query}"\n\n{INSTRUCTIONS}\n\nRecords:\n{listing}'},
        ],
    )
    data = json.loads(resp.choices[0].message.content or "{}")
    return {int(i) for i in data.get("relevant", [])}


def judge(conn, query: str, candidates: list[dict]) -> dict[int, bool]:
    """Map candidate id -> is-relevant. Cached per (query, chunk)."""
    if not candidates:
        return {}

    qh = hashlib.sha256(f"{config.CHAT_DEPLOYMENT}|{query.strip().lower()}".encode()).hexdigest()
    cached, todo = {}, []
    for c in candidates:
        row = conn.execute(
            "SELECT relevant FROM rerank_cache WHERE qhash=? AND chunk_id=?", (qh, c["id"])
        ).fetchone()
        (cached.__setitem__(c["id"], bool(row["relevant"])) if row else todo.append(c))
    if not todo:
        return cached

    if config.OFFLINE_EMBED:
        verdicts = _stub(query, todo)
    else:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            azure_endpoint=config.AZURE_ENDPOINT,
            api_key=config.AZURE_API_KEY,
            api_version=config.AZURE_API_VERSION,
        )
        keep: set[int] = set()
        for i in range(0, len(todo), config.RERANK_BATCH):
            keep |= _ask(client, query, todo[i:i + config.RERANK_BATCH])
        verdicts = {c["id"]: (c["id"] in keep) for c in todo}

    conn.executemany(
        "INSERT OR REPLACE INTO rerank_cache(qhash, chunk_id, relevant) VALUES (?,?,?)",
        [(qh, cid, int(v)) for cid, v in verdicts.items()],
    )
    conn.commit()
    return {**cached, **verdicts}
