"""
Financial Research Agent — Streamlit Interface
本地运行，调用 Vertex AI Vector Search + Gemini 3.1 Pro
"""

import streamlit as st
import json
import os
import sys

# ── 页面配置 ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Research Agent",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Financial Research Agent")
st.caption("Powered by Gemini 3.1 Pro · Vertex AI Vector Search · SEC EDGAR Data")

# ── 侧边栏：公司选择 ────────────────────────────────────────────
with st.sidebar:
    st.header("🏢 Company Filter")
    ticker_options = ["All Companies", "AAPL", "NVDA", "TSLA", "MSFT", "RIVN"]
    selected_ticker = st.selectbox("Select company (optional)", ticker_options)
    ticker = None if selected_ticker == "All Companies" else selected_ticker

    st.divider()
    st.header("⚙️ Settings")
    top_k = st.slider("Number of sources to retrieve", 3, 10, 5)

    st.divider()
    st.header("🎯 Risk Mode")
    risk_mode = st.radio(
        "Retrieval strategy:",
        ["Balanced", "High Recall (don't miss risks)", "High Precision (only certain risks)"],
        index=0,
        help="High Recall = 宁可误报也不漏报（适合风控）\nHigh Precision = 只报告确定的风险（适合投资建议）"
    )
    alpha_map = {"Balanced": 0.6, "High Recall (don't miss risks)": 0.35, "High Precision (only certain risks)": 0.8}
    retrieval_alpha = alpha_map[risk_mode]
    st.caption(f"Dense/Sparse weight: {retrieval_alpha:.0%} / {1-retrieval_alpha:.0%}")

    st.divider()
    st.markdown("**Data Coverage**")
    st.markdown("- 🍎 Apple (2021–2025)")
    st.markdown("- 🟢 NVIDIA (2022–2026)")
    st.markdown("- ⚡ Tesla (2022–2025)")
    st.markdown("- 🪟 Microsoft (2021–2025)")
    st.markdown("- 🚛 Rivian (2021–2025)")

    st.divider()
    st.markdown("**Tech Stack**")
    st.markdown("Gemini 3.1 Pro · Vertex AI · SEC EDGAR")

    st.divider()
    st.header("🧪 Distillation eval (offline)")
    _eval_path = os.path.join(os.path.dirname(__file__), "evaluation_results_v2.json")
    if os.path.exists(_eval_path):
        import json as _json
        _eval = _json.load(open(_eval_path))
        base_f1 = _eval["results"]["base"]["bertscore_f1"]
        ft_f1   = _eval["results"]["v2"]["bertscore_f1"]
        delta   = _eval["deltas"]["base_to_v2_pct"]
        n       = _eval["test_set_size"]

        col_a, col_b = st.columns(2)
        col_a.metric("Base Gemma 2 27B", f"{base_f1:.4f}", help="BERTScore F1, no adapter")
        col_b.metric("+ SEC LoRA",       f"{ft_f1:.4f}",   delta=f"+{delta:.2f}%",
                     help="BERTScore F1 with LoRA rank=8 adapter trained on distilled SEC QA")

        st.caption(f"BERTScore F1 on n={n} held-out SEC QA items (TPU-side eval)")
    else:
        st.caption("Run `compute_bertscore_v2.py` to populate.")

# ── 示例问题 ────────────────────────────────────────────────────
st.subheader("💡 Try these questions")
example_cols = st.columns(3)
examples = [
    "What are NVIDIA's biggest risks from export controls and how have they evolved?",
    "Compare Apple and Microsoft's AI strategy based on their annual reports.",
    "How has Rivian's path to profitability changed since their IPO?",
]
for i, (col, ex) in enumerate(zip(example_cols, examples)):
    if col.button(ex, key=f"ex_{i}", use_container_width=True):
        st.session_state["question"] = ex

# ── 主输入框 ────────────────────────────────────────────────────
question = st.text_area(
    "Ask a financial question:",
    value=st.session_state.get("question", ""),
    height=80,
    placeholder="e.g. What are the key risks for NVIDIA in 2025? How has Apple's revenue grown?"
)

run_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

# ── 加载 RAG Agent ──────────────────────────────────────────────
@st.cache_resource
def load_agent():
    try:
        from google import genai
        from google.genai import types
        from google.cloud import aiplatform
        from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import Namespace

        PROJECT_ID = "project-1faae058-abd0-4492-82f"
        LOCATION = "us-central1"
        INDEX_ENDPOINT_ID = "2952648316438970368"
        DEPLOYED_INDEX_ID = "sec_financial_deployed"

        embed_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
        gen_client = genai.Client(vertexai=True, project=PROJECT_ID, location="global")
        aiplatform.init(project=PROJECT_ID, location=LOCATION)

        return embed_client, gen_client, aiplatform, Namespace, types, PROJECT_ID, LOCATION, INDEX_ENDPOINT_ID, DEPLOYED_INDEX_ID
    except Exception as e:
        return None, None, None, None, None, None, None, None, str(e)

def run_rag(question, ticker, top_k):
    embed_client, gen_client, aiplatform, Namespace, types, PROJECT_ID, LOCATION, INDEX_ENDPOINT_ID, DEPLOYED_INDEX_ID = load_agent()

    if embed_client is None:
        return None, None, f"Failed to load agent: {DEPLOYED_INDEX_ID}"

    # Step 1: Embed query
    result = embed_client.models.embed_content(
        model="gemini-embedding-001",
        contents=question,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    query_vector = result.embeddings[0].values

    # Step 2: Retrieve from Vector Search
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

    # Step 3: Load context text
    summaries_dir = os.path.join(os.path.dirname(__file__), "summaries")
    context_parts = []
    sources = []

    for hit in hits:
        doc_id = hit.id
        parts = doc_id.split("_")
        if parts[0] == "summary" and len(parts) >= 3:
            t, y = parts[1], parts[2]
            fpath = os.path.join(summaries_dir, f"{t}_{y}_summary.json")
            if os.path.exists(fpath):
                with open(fpath) as f:
                    s = json.load(f)
                context_parts.append(
                    f"[{t} FY{y}]\n"
                    f"Risks: {'; '.join(s.get('top_risks', []))}\n"
                    f"Highlights: {'; '.join(s.get('strategic_highlights', []))}\n"
                    f"MDA: {s.get('mda_summary', '')}\n"
                    f"Analyst Note: {s.get('analyst_note', '')}"
                )
                sources.append({"ticker": t, "year": y, "score": round(hit.distance, 3)})

    context = "\n\n---\n\n".join(context_parts)

    # Step 4: Generate answer
    company_ctx = f"Focus: {ticker.upper()}" if ticker else "Multi-company analysis"
    prompt = f"""You are a senior Wall Street equity research analyst.

{company_ctx}

Use the retrieved SEC filing context below to answer the question. Be specific, cite figures when available, use structured formatting with headers.

=== CONTEXT ===
{context}
=== END CONTEXT ===

Question: {question}"""

    resp = gen_client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2),
    )
    return resp.text, sources, None, None

# ── 运行查询 ────────────────────────────────────────────────────
if run_btn and question.strip():
    with st.spinner("🔍 Retrieving from SEC filings... 📊 Generating analysis..."):
        answer, sources, error, _ = run_rag(question, ticker, top_k)

    if error:
        st.error(f"Error: {error}")
    else:
        st.divider()
        st.subheader("📋 Analysis")
        st.markdown(answer)

        if sources:
            st.divider()
            st.subheader("📂 Sources Retrieved")
            cols = st.columns(len(sources))
            for col, src in zip(cols, sources):
                col.metric(
                    label=f"{src['ticker']} FY{src['year']}",
                    value=f"Score: {src['score']}"
                )

elif run_btn:
    st.warning("Please enter a question.")

# ── Footer ──────────────────────────────────────────────────────
st.divider()
st.caption("Data source: SEC EDGAR · Model: Gemini 3.1 Pro Preview via Vertex AI · Vector DB: Vertex AI Vector Search")
