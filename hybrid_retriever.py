"""
Hybrid Retrieval: Dense (Vertex AI Vector Search) + Sparse (BM25)
理论依据：Dense retrieval 捕获语义相似性，BM25 精确匹配金融专业术语
如 "EBITDA", "non-GAAP", "operating leverage", "covenant breach" 等
"""

import json
import os
import math
from rank_bm25 import BM25Okapi
from google import genai
from google.genai import types
from google.cloud import aiplatform
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import Namespace

PROJECT_ID = "project-1faae058-abd0-4492-82f"
LOCATION = "us-central1"
INDEX_ENDPOINT_ID = "2952648316438970368"
DEPLOYED_INDEX_ID = "sec_financial_deployed"
SUMMARIES_DIR = os.path.join(os.path.dirname(__file__), "summaries")

embed_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
gen_client = genai.Client(vertexai=True, project=PROJECT_ID, location="global")
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# ── BM25 索引（从摘要文件构建）──────────────────────────────────
def build_bm25_index():
    """加载所有摘要，构建 BM25 稀疏检索索引"""
    corpus = []
    doc_ids = []

    for fname in sorted(os.listdir(SUMMARIES_DIR)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(SUMMARIES_DIR, fname)) as f:
            s = json.load(f)

        ticker = s.get("ticker", "")
        year = s.get("fiscal_year", "")
        text = " ".join([
            ticker, year,
            " ".join(s.get("top_risks", [])),
            " ".join(s.get("strategic_highlights", [])),
            s.get("mda_summary", ""),
            s.get("analyst_note", ""),
        ])
        tokens = text.lower().split()
        corpus.append(tokens)
        doc_ids.append(f"summary_{ticker}_{year}")

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

    # 归一化为 0-1 分数
    scores = {}
    for i, hit in enumerate(hits):
        if hit.id.startswith("summary_"):
            scores[hit.id] = 1.0 - (i / len(hits))  # 排名越前分越高
    return scores

# ── Sparse Retrieval（BM25）────────────────────────────────────
def sparse_retrieve(question, ticker=None, top_k=5):
    tokens = question.lower().split()
    bm25_scores = _bm25_index.get_scores(tokens)

    doc_scores = {}
    for idx, score in enumerate(bm25_scores):
        doc_id = _bm25_doc_ids[idx]
        # 如果指定 ticker，过滤
        if ticker and f"summary_{ticker.upper()}_" not in doc_id:
            continue
        if score > 0:
            doc_scores[doc_id] = score

    # 归一化
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

# ── 加载上下文文本 ───────────────────────────────────────────────
def load_context(ranked_ids):
    context_parts = []
    sources = []

    for doc_id, score in ranked_ids:
        parts = doc_id.split("_")
        if parts[0] == "summary" and len(parts) >= 3:
            t, y = parts[1], parts[2]
            fpath = os.path.join(SUMMARIES_DIR, f"{t}_{y}_summary.json")
            if os.path.exists(fpath):
                with open(fpath) as f:
                    s = json.load(f)
                context_parts.append(
                    f"[{t} FY{y} — Hybrid Score: {score:.3f}]\n"
                    f"Risks: {'; '.join(s.get('top_risks', []))}\n"
                    f"Highlights: {'; '.join(s.get('strategic_highlights', []))}\n"
                    f"MDA: {s.get('mda_summary', '')}\n"
                    f"Analyst Note: {s.get('analyst_note', '')}"
                )
                sources.append({"ticker": t, "year": y, "score": round(score, 3)})

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
    for s in sources:
        print(f"  {s['ticker']} FY{s['year']} — hybrid score: {s['score']}")
    print(f"\n{resp.text}")
    return resp.text, sources


if __name__ == "__main__":
    # 测试：金融专业术语查询（用 keyword-heavy alpha=0.3）
    ask("What is NVIDIA's non-GAAP operating margin and EBITDA trend?",
        ticker="NVDA", alpha=0.3)

    # 测试：语义理解查询（用 semantic-heavy alpha=0.7）
    ask("How exposed is Rivian to macroeconomic headwinds?",
        ticker="RIVN", alpha=0.7)
