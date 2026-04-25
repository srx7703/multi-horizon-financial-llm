"""
Hybrid Retrieval: Dense (Vertex AI Vector Search) + Sparse (BM25)
理论依据：Dense retrieval 捕获语义相似性，BM25 精确匹配金融专业术语
如 "EBITDA", "non-GAAP", "operating leverage", "covenant breach" 等
"""

import json
import os
from pathlib import Path
from rank_bm25 import BM25Okapi
from google import genai
from google.genai import types
from google.cloud import aiplatform
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import Namespace

PROJECT_ID = "project-1faae058-abd0-4492-82f"
LOCATION = "us-central1"
INDEX_ENDPOINT_ID = "2952648316438970368"
DEPLOYED_INDEX_ID = "sec_financial_deployed"

ROOT = Path(__file__).parent
SUMMARY_DIRS = {
    "10k": ROOT / "summaries",
    "10q": ROOT / "summaries_10q",
    "8k":  ROOT / "summaries_8k",
}

embed_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
gen_client = genai.Client(vertexai=True, project=PROJECT_ID, location="global")
aiplatform.init(project=PROJECT_ID, location=LOCATION)


def _doc_text(ftype: str, s: dict) -> str:
    if ftype == "10k":
        return " ".join(filter(None, [
            s.get("ticker", ""), str(s.get("fiscal_year", "")),
            " ".join(s.get("top_risks", [])),
            " ".join(s.get("strategic_highlights", [])),
            s.get("mda_summary", ""),
            s.get("analyst_note", ""),
        ]))
    if ftype == "10q":
        return " ".join(filter(None, [
            s.get("ticker", ""), s.get("fiscal_period", ""), s.get("period_end", ""),
            s.get("revenue", ""), s.get("net_income", ""),
            " ".join(s.get("key_metrics", [])),
            " ".join(s.get("qoq_changes", [])),
            " ".join(s.get("new_risks", [])),
            s.get("management_tone", ""), s.get("analyst_note", ""),
        ]))
    return " ".join(filter(None, [
        s.get("ticker", ""), s.get("filing_date", ""), s.get("event_type", ""),
        s.get("headline", ""), s.get("investor_impact", ""),
        " ".join(map(str, s.get("key_figures", []))),
    ]))


def _doc_id(ftype: str, fp: Path, s: dict) -> str:
    ticker = (s.get("ticker") or "").upper()
    stem = fp.stem
    if ftype == "10k":
        year = str(s.get("fiscal_year") or stem.split("_")[1])
        return f"10k_{ticker}_{year}"
    if ftype == "10q":
        date = s.get("period_end") or stem.split("_")[1]
        return f"10q_{ticker}_{date}"
    parts = stem.split("_")
    return f"8k_{ticker}_{parts[1]}_{parts[2]}"


# ── BM25 索引（10-K + 10-Q + 8-K 全量）──────────────────────────
def build_bm25_index():
    corpus, doc_ids = [], []
    for ftype, d in SUMMARY_DIRS.items():
        if not d.is_dir():
            continue
        for fp in sorted(d.glob("*.json")):
            try:
                s = json.loads(fp.read_text())
            except Exception:
                continue
            if not s.get("ticker"):
                continue
            corpus.append(_doc_text(ftype, s).lower().split())
            doc_ids.append(_doc_id(ftype, fp, s))
    return BM25Okapi(corpus), doc_ids


_bm25_index, _bm25_doc_ids = build_bm25_index()

# ── Dense Retrieval（向量检索）──────────────────────────────────
def dense_retrieve(question, ticker=None, top_k=5):
    result = embed_client.models.embed_content(
        model="gemini-embedding-001",
        contents=question,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    query_vector = result.embeddings[0].values

    endpoint = aiplatform.MatchingEngineIndexEndpoint(
        f"projects/927558868397/locations/{LOCATION}/indexEndpoints/{INDEX_ENDPOINT_ID}"
    )
    filter_arg = [Namespace(name="ticker", allow_tokens=[ticker.upper()])] if ticker else None
    response = endpoint.find_neighbors(
        deployed_index_id=DEPLOYED_INDEX_ID,
        queries=[query_vector],
        num_neighbors=top_k,
        filter=filter_arg,
    )
    hits = response[0] if response else []

    # 归一化为 0-1 分数（保留所有 type 前缀的 doc_id）
    scores = {}
    for i, hit in enumerate(hits):
        scores[hit.id] = 1.0 - (i / max(len(hits), 1))
    return scores


def _doc_id_matches_ticker(doc_id: str, ticker: str) -> bool:
    """doc_id schemes: 10k_<T>_<Y>, 10q_<T>_<D>, 8k_<T>_<D>_<I>"""
    parts = doc_id.split("_")
    return len(parts) >= 2 and parts[1] == ticker.upper()


# ── Sparse Retrieval（BM25）────────────────────────────────────
def sparse_retrieve(question, ticker=None, top_k=5):
    tokens = question.lower().split()
    bm25_scores = _bm25_index.get_scores(tokens)

    doc_scores = {}
    for idx, score in enumerate(bm25_scores):
        doc_id = _bm25_doc_ids[idx]
        if ticker and not _doc_id_matches_ticker(doc_id, ticker):
            continue
        if score > 0:
            doc_scores[doc_id] = score

    if doc_scores:
        max_score = max(doc_scores.values())
        doc_scores = {k: v / max_score for k, v in doc_scores.items()}

    return doc_scores

# ── Hybrid Fusion（RRF 算法合并两路结果）─────────────────────────
def hybrid_retrieve(question, ticker=None, top_k=5, alpha=0.6):
    """
    alpha: Dense 权重（0.6 = 偏向语义理解；0.4 = 偏向精确匹配）
    金融专业术语查询建议降低 alpha（如 0.3），通用问题用 0.6
    """
    dense_scores = dense_retrieve(question, ticker, top_k * 2)
    sparse_scores = sparse_retrieve(question, ticker, top_k * 2)

    all_ids = set(dense_scores) | set(sparse_scores)
    fused = {}
    for doc_id in all_ids:
        d = dense_scores.get(doc_id, 0)
        s = sparse_scores.get(doc_id, 0)
        fused[doc_id] = alpha * d + (1 - alpha) * s

    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return ranked

# ── 加载上下文文本（10-K / 10-Q / 8-K 三种 schema）────────────────
def _resolve(doc_id: str):
    parts = doc_id.split("_")
    if len(parts) < 3:
        return None
    ftype, t = parts[0], parts[1]
    if ftype == "10k":
        year = parts[2]
        fp = SUMMARY_DIRS["10k"] / f"{t}_{year}_summary.json"
        label = f"FY{year}"
    elif ftype == "10q":
        date = parts[2]
        fp = SUMMARY_DIRS["10q"] / f"{t}_{date}_10Q.json"
        label = date
    elif ftype == "8k" and len(parts) >= 4:
        date, fid = parts[2], parts[3]
        fp = SUMMARY_DIRS["8k"] / f"{t}_{date}_{fid}_8K.json"
        label = date
    else:
        return None
    if not fp.exists():
        return None
    return ftype, t, label, json.loads(fp.read_text())


def load_context(ranked_ids):
    context_parts = []
    sources = []

    for doc_id, score in ranked_ids:
        resolved = _resolve(doc_id)
        if resolved is None:
            continue
        ftype, t, label, s = resolved
        if ftype == "10k":
            block = (
                f"[{t} 10-K {label} — Hybrid Score: {score:.3f}]\n"
                f"Risks: {'; '.join(s.get('top_risks', []))}\n"
                f"Highlights: {'; '.join(s.get('strategic_highlights', []))}\n"
                f"MDA: {s.get('mda_summary', '')}\n"
                f"Analyst Note: {s.get('analyst_note', '')}"
            )
        elif ftype == "10q":
            block = (
                f"[{t} 10-Q {s.get('fiscal_period', label)} — Hybrid Score: {score:.3f}]\n"
                f"Revenue: {s.get('revenue', '')} · Net income: {s.get('net_income', '')}\n"
                f"Key metrics: {'; '.join(s.get('key_metrics', []))}\n"
                f"QoQ: {'; '.join(s.get('qoq_changes', []))}\n"
                f"Tone: {s.get('management_tone', '')}\n"
                f"Analyst Note: {s.get('analyst_note', '')}"
            )
        else:
            block = (
                f"[{t} 8-K {label} — {s.get('event_type', '')} — Hybrid Score: {score:.3f}]\n"
                f"Headline: {s.get('headline', '')}\n"
                f"Materiality: {s.get('materiality', '')}\n"
                f"Investor impact: {s.get('investor_impact', '')}"
            )
        context_parts.append(block)
        sources.append({"ticker": t, "filing_type": ftype, "label": label, "score": round(score, 3)})

    return "\n\n---\n\n".join(context_parts), sources

# ── 主入口 ───────────────────────────────────────────────────────
def ask(question, ticker=None, alpha=0.6):
    print(f"\nQuestion: {question}")
    print(f"Mode: alpha={alpha} ({'semantic-heavy' if alpha > 0.5 else 'keyword-heavy'})")

    ranked = hybrid_retrieve(question, ticker=ticker, alpha=alpha)
    context, sources = load_context(ranked)

    prompt = f"""You are a senior Wall Street equity analyst.
{'Focus: ' + ticker.upper() if ticker else 'Multi-company analysis'}

Use the retrieved SEC filing context to answer. Be specific and structured.

=== CONTEXT ===
{context}
=== END ===

Question: {question}"""

    resp = gen_client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2),
    )
    print("\nSources retrieved:")
    type_label = {"10k": "10-K", "10q": "10-Q", "8k": "8-K"}
    for s in sources:
        print(f"  {s['ticker']} {type_label[s['filing_type']]} {s['label']} — hybrid score: {s['score']}")
    print(f"\n{resp.text}")
    return resp.text, sources


if __name__ == "__main__":
    # 测试：金融专业术语查询（用 keyword-heavy alpha=0.3）
    ask("What is NVIDIA's non-GAAP operating margin and EBITDA trend?",
        ticker="NVDA", alpha=0.3)

    # 测试：语义理解查询（用 semantic-heavy alpha=0.7）
    ask("How does AAPL describe macroeconomic headwinds in recent 10-Q filings?",
        ticker="AAPL", alpha=0.7)
