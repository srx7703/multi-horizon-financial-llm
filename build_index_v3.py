"""Re-build Vertex AI Vector Search index with full SEC coverage.

Indexes all 10-K + 10-Q + 8-K summaries with consistent doc_id schemes:
  10-K: 10k_<TICKER>_<YEAR>
  10-Q: 10q_<TICKER>_<DATE>      (date = period_end, e.g. 2025-06-28)
  8-K:  8k_<TICKER>_<DATE>_<ID>  (matches filename ordinal)

Each datapoint carries restricts on `ticker` and `filing_type` so the demo can
filter by both. Uses BATCH_UPDATE flow: writes JSONL to GCS, then triggers
`index.update_embeddings(..., is_complete_overwrite=True)` which fully replaces
prior content (including the legacy 5 `summary_*` entries).

Run:
    python build_index_v3.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from google import genai
from google.genai import types
from google.cloud import aiplatform, storage

PROJECT_ID = "project-1faae058-abd0-4492-82f"
LOCATION = "us-central1"
INDEX_ID = "8999663202044215296"
GCS_BUCKET = "sec-financial-agent-p1faae058"
GCS_PREFIX = "vector_search_index_v3"

ROOT = Path(__file__).parent
DIRS = {
    "10k": ROOT / "summaries",
    "10q": ROOT / "summaries_10q",
    "8k": ROOT / "summaries_8k",
}

CACHE_FILE = ROOT / ".embed_cache_v3.jsonl"


def text_for_10k(s: dict) -> str:
    return " ".join(filter(None, [
        s.get("ticker", ""), s.get("fiscal_year", ""),
        " ".join(s.get("top_risks", [])),
        " ".join(s.get("strategic_highlights", [])),
        s.get("mda_summary", ""),
        s.get("analyst_note", ""),
    ]))


def text_for_10q(s: dict) -> str:
    return " ".join(filter(None, [
        s.get("ticker", ""), s.get("fiscal_period", ""), s.get("period_end", ""),
        s.get("revenue", ""), s.get("net_income", ""),
        " ".join(s.get("key_metrics", [])),
        " ".join(s.get("qoq_changes", [])),
        " ".join(s.get("new_risks", [])),
        s.get("management_tone", ""),
        s.get("analyst_note", ""),
    ]))


def text_for_8k(s: dict) -> str:
    return " ".join(filter(None, [
        s.get("ticker", ""), s.get("filing_date", ""), s.get("event_type", ""),
        s.get("headline", ""), s.get("investor_impact", ""),
        " ".join(map(str, s.get("key_figures", []))),
    ]))


def collect_docs() -> list[dict]:
    docs = []
    for ftype, dir_path in DIRS.items():
        for fp in sorted(dir_path.glob("*.json")):
            s = json.loads(fp.read_text())
            ticker = (s.get("ticker") or "").upper()
            if not ticker:
                continue
            stem = fp.stem
            if ftype == "10k":
                year = str(s.get("fiscal_year") or stem.split("_")[1])
                doc_id = f"10k_{ticker}_{year}"
                text = text_for_10k(s)
                meta = {"year": year}
            elif ftype == "10q":
                date = s.get("period_end") or stem.split("_")[1]
                doc_id = f"10q_{ticker}_{date}"
                text = text_for_10q(s)
                meta = {"period_end": date, "fiscal_period": s.get("fiscal_period", "")}
            else:
                parts = stem.split("_")
                date, fid = parts[1], parts[2]
                doc_id = f"8k_{ticker}_{date}_{fid}"
                text = text_for_8k(s)
                meta = {"filing_date": date, "event_type": s.get("event_type", "")}
            docs.append({"id": doc_id, "text": text, "ticker": ticker, "type": ftype, "meta": meta})
    return docs


def load_cache() -> dict[str, list[float]]:
    if not CACHE_FILE.exists():
        return {}
    cache = {}
    with CACHE_FILE.open() as f:
        for line in f:
            row = json.loads(line)
            cache[row["id"]] = row["embedding"]
    return cache


def append_cache(doc_id: str, embedding: list[float]) -> None:
    with CACHE_FILE.open("a") as f:
        f.write(json.dumps({"id": doc_id, "embedding": embedding}) + "\n")


def embed_all(docs: list[dict]) -> dict[str, list[float]]:
    cache = load_cache()
    print(f"loaded {len(cache)} cached embeddings")
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    n_new = 0
    for i, d in enumerate(docs):
        if d["id"] in cache:
            continue
        # gemini-embedding-001, 3072-dim, RETRIEVAL_DOCUMENT
        for attempt in range(3):
            try:
                r = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=d["text"],
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
                )
                vec = list(r.embeddings[0].values)
                cache[d["id"]] = vec
                append_cache(d["id"], vec)
                n_new += 1
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"  retry {attempt+1} after {wait}s: {e}")
                time.sleep(wait)
        else:
            raise RuntimeError(f"failed to embed {d['id']}")
        if (i + 1) % 25 == 0:
            print(f"  embedded {i+1}/{len(docs)} (new this run: {n_new})")
    print(f"done: {len(cache)}/{len(docs)} embedded ({n_new} new)")
    return cache


def write_jsonl_to_gcs(docs: list[dict], embeddings: dict[str, list[float]]) -> str:
    local_jsonl = ROOT / "_index_payload.json"
    with local_jsonl.open("w") as f:
        for d in docs:
            row = {
                "id": d["id"],
                "embedding": embeddings[d["id"]],
                "restricts": [
                    {"namespace": "ticker", "allow": [d["ticker"]]},
                    {"namespace": "filing_type", "allow": [d["type"]]},
                ],
            }
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(docs)} datapoints to {local_jsonl}")

    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(GCS_BUCKET)
    blob_path = f"{GCS_PREFIX}/data.json"
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(str(local_jsonl))
    gcs_uri = f"gs://{GCS_BUCKET}/{GCS_PREFIX}/"
    print(f"uploaded to {gcs_uri}")
    return gcs_uri


def trigger_index_update(gcs_uri: str) -> None:
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    index = aiplatform.MatchingEngineIndex(INDEX_ID)
    print(f"triggering update_embeddings (overwrite=True) on index {INDEX_ID}")
    index.update_embeddings(
        contents_delta_uri=gcs_uri,
        is_complete_overwrite=True,
    )
    print("update_embeddings call returned. Background re-index typically takes 10-30 minutes.")
    print("Check status: gcloud ai indexes describe", INDEX_ID, "--project", PROJECT_ID, "--region", LOCATION)


def main():
    docs = collect_docs()
    by_type = {}
    by_ticker = set()
    for d in docs:
        by_type[d["type"]] = by_type.get(d["type"], 0) + 1
        by_ticker.add(d["ticker"])
    print(f"collected {len(docs)} docs across {len(by_ticker)} tickers")
    print(f"by type: {by_type}")

    embeddings = embed_all(docs)
    gcs_uri = write_jsonl_to_gcs(docs, embeddings)
    trigger_index_update(gcs_uri)


if __name__ == "__main__":
    main()
