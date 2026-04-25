"""
Financial Research Agent — Streamlit Interface
本地运行，调用 Vertex AI Vector Search + Gemini 3.1 Pro
"""

import streamlit as st
import json
import os
import sys
from pathlib import Path

# ── 页面配置 ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Research Agent",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Financial Research Agent")
st.caption("Powered by Gemini 3.1 Pro · Vertex AI Vector Search · SEC EDGAR (10-K / 10-Q / 8-K)")

# ── 数据覆盖（从硬盘扫描，不再硬编码）────────────────────────────
ROOT = Path(__file__).parent
SUMMARY_DIRS = {
    "10-K": ROOT / "summaries",
    "10-Q": ROOT / "summaries_10q",
    "8-K":  ROOT / "summaries_8k",
}


@st.cache_data
def scan_coverage():
    """Walk summaries dirs once; return per-ticker filing-type counts."""
    coverage = {}
    totals = {t: 0 for t in SUMMARY_DIRS}
    for ftype, d in SUMMARY_DIRS.items():
        if not d.is_dir():
            continue
        for fp in d.glob("*.json"):
            try:
                ticker = json.loads(fp.read_text()).get("ticker", "").upper()
            except Exception:
                continue
            if not ticker:
                continue
            coverage.setdefault(ticker, {t: 0 for t in SUMMARY_DIRS})[ftype] += 1
            totals[ftype] += 1
    return coverage, totals


COVERAGE, TOTALS = scan_coverage()
ALL_TICKERS = sorted(COVERAGE.keys())

# ── 侧边栏 ────────────────────────────────────────────────────
with st.sidebar:
    st.header("🏢 Company Filter")
    selected_ticker = st.selectbox(
        f"Select company ({len(ALL_TICKERS)} indexed)",
        ["All Companies"] + ALL_TICKERS,
    )
    ticker = None if selected_ticker == "All Companies" else selected_ticker

    filing_type = st.selectbox(
        "Filing type",
        ["All filings", "10-K (annual)", "10-Q (quarterly)", "8-K (event)"],
    )
    filing_type_token = {
        "All filings": None,
        "10-K (annual)": "10k",
        "10-Q (quarterly)": "10q",
        "8-K (event)": "8k",
    }[filing_type]

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
    st.caption(
        f"{len(ALL_TICKERS)} S&P 500 tickers · "
        f"{TOTALS['10-K']} 10-Ks · {TOTALS['10-Q']} 10-Qs · {TOTALS['8-K']} 8-Ks"
    )
    if ticker and ticker in COVERAGE:
        c = COVERAGE[ticker]
        st.caption(
            f"**{ticker}**: {c['10-K']} 10-K · {c['10-Q']} 10-Q · {c['8-K']} 8-K"
        )

    st.divider()
    st.markdown("**Tech Stack**")
    st.markdown("Gemini 3.1 Pro · Vertex AI · SEC EDGAR")

    st.divider()
    st.header("🧪 LoRA fine-tune eval (offline)")
    _phase2_path = ROOT / "evaluation_results_phase2.json"
    _phase1_path = ROOT / "evaluation_results_v2.json"

    if _phase2_path.exists():
        _eval = json.load(open(_phase2_path))
        col_a, col_b = st.columns(2)
        col_a.metric(
            "Gemma 2 27B + LoRA",
            f"{_eval['results']['v2']['bertscore_f1']:.4f}",
            delta=f"+{_eval['deltas']['gemma2_base_to_v2_pct']:.2f}%",
            help="Phase 1: vs base Gemma 2 27B",
        )
        col_b.metric(
            "Gemma 4 31B + LoRA",
            f"{_eval['results']['gemma4_v2g4']['bertscore_f1']:.4f}",
            delta=f"+{_eval['deltas']['gemma4_base_to_v2g4_pct']:.2f}%",
            help="Phase 2: vs base Gemma 4 31B",
        )
        st.caption(f"BERTScore F1, n={_eval['test_set_size']} held-out · paired t-test, p<0.001")
    elif _phase1_path.exists():
        _eval = json.load(open(_phase1_path))
        col_a, col_b = st.columns(2)
        col_a.metric("Base Gemma 2 27B", f"{_eval['results']['base']['bertscore_f1']:.4f}")
        col_b.metric("+ SEC LoRA", f"{_eval['results']['v2']['bertscore_f1']:.4f}",
                     delta=f"+{_eval['deltas']['base_to_v2_pct']:.2f}%")
        st.caption(f"BERTScore F1, n={_eval['test_set_size']} held-out items")
    else:
        st.caption("Run eval scripts to populate.")

# ── 示例问题 ────────────────────────────────────────────────────
st.subheader("💡 Try these questions")
example_cols = st.columns(3)
examples = [
    "What are NVIDIA's biggest risks from export controls and how have they evolved across recent 10-Ks?",
    "Compare Q3 2025 services-revenue growth narratives for Apple and Microsoft.",
    "Summarize the most material 8-K events at JPM, GS, and BAC over the past 6 months.",
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

def edgar_url(ticker: str, filing_type: str) -> str:
    """Deep-link to a ticker's filings of a given type on SEC EDGAR full-text search."""
    type_map = {"10k": "10-K", "10q": "10-Q", "8k": "8-K"}
    return (
        "https://www.sec.gov/cgi-bin/browse-edgar"
        f"?action=getcompany&CIK={ticker}&type={type_map.get(filing_type, '')}"
        "&dateb=&owner=include&count=40"
    )


def resolve_summary(doc_id: str):
    """Map a Vector Search doc_id to its on-disk summary JSON.

    doc_id schemes:
      10k_<TICKER>_<YEAR>
      10q_<TICKER>_<DATE>
      8k_<TICKER>_<DATE>_<ID>
    Returns (filing_type, ticker, label, summary_dict) or None.
    """
    parts = doc_id.split("_")
    if len(parts) < 3:
        return None
    ftype, ticker = parts[0], parts[1]
    base = os.path.dirname(__file__)
    if ftype == "10k":
        year = parts[2]
        fpath = os.path.join(base, "summaries", f"{ticker}_{year}_summary.json")
        label = f"FY{year}"
    elif ftype == "10q":
        date = parts[2]
        fpath = os.path.join(base, "summaries_10q", f"{ticker}_{date}_10Q.json")
        label = date
    elif ftype == "8k":
        date, fid = parts[2], parts[3]
        fpath = os.path.join(base, "summaries_8k", f"{ticker}_{date}_{fid}_8K.json")
        label = date
    else:
        return None
    if not os.path.exists(fpath):
        return None
    with open(fpath) as f:
        return ftype, ticker, label, json.load(f)


def context_block(ftype: str, ticker: str, label: str, s: dict) -> str:
    """Format a per-doc context block, schema-aware by filing type."""
    if ftype == "10k":
        return (
            f"[{ticker} 10-K {label}]\n"
            f"Risks: {'; '.join(s.get('top_risks', []))}\n"
            f"Highlights: {'; '.join(s.get('strategic_highlights', []))}\n"
            f"MDA: {s.get('mda_summary', '')}\n"
            f"Analyst Note: {s.get('analyst_note', '')}"
        )
    if ftype == "10q":
        return (
            f"[{ticker} 10-Q {s.get('fiscal_period', label)}]\n"
            f"Revenue: {s.get('revenue', '')} · Net income: {s.get('net_income', '')}\n"
            f"Key metrics: {'; '.join(s.get('key_metrics', []))}\n"
            f"QoQ changes: {'; '.join(s.get('qoq_changes', []))}\n"
            f"New risks: {'; '.join(s.get('new_risks', []))}\n"
            f"Tone: {s.get('management_tone', '')}\n"
            f"Analyst Note: {s.get('analyst_note', '')}"
        )
    return (
        f"[{ticker} 8-K {label} — {s.get('event_type', '')}]\n"
        f"Headline: {s.get('headline', '')}\n"
        f"Materiality: {s.get('materiality', '')}\n"
        f"Investor impact: {s.get('investor_impact', '')}"
    )


def run_rag(question, ticker, top_k, filing_type_token=None):
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

    # Step 2: Retrieve from Vector Search (filter by ticker and/or filing type)
    endpoint = aiplatform.MatchingEngineIndexEndpoint(
        f"projects/927558868397/locations/{LOCATION}/indexEndpoints/{INDEX_ENDPOINT_ID}"
    )
    filter_arg = []
    if ticker:
        filter_arg.append(Namespace(name="ticker", allow_tokens=[ticker.upper()]))
    if filing_type_token:
        filter_arg.append(Namespace(name="filing_type", allow_tokens=[filing_type_token]))
    response = endpoint.find_neighbors(
        deployed_index_id=DEPLOYED_INDEX_ID,
        queries=[query_vector],
        num_neighbors=top_k,
        filter=filter_arg or None,
    )
    hits = response[0] if response else []

    # Step 3: Load context text per filing type
    context_parts = []
    sources = []
    for hit in hits:
        resolved = resolve_summary(hit.id)
        if resolved is None:
            continue
        ftype, t, label, s = resolved
        context_parts.append(context_block(ftype, t, label, s))
        sources.append({
            "doc_id": hit.id,
            "ticker": t,
            "filing_type": ftype,
            "label": label,
            "score": round(hit.distance, 3),
            "summary": s,
        })

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
        answer, sources, error, _ = run_rag(question, ticker, top_k, filing_type_token)

    if error:
        st.error(f"Error: {error}")
    else:
        st.divider()
        st.subheader("📋 Analysis")
        with st.container(border=True):
            st.markdown(answer)

        if sources:
            st.divider()
            st.subheader(f"📂 Sources Retrieved ({len(sources)})")
            type_label = {"10k": "10-K", "10q": "10-Q", "8k": "8-K"}
            type_icon = {"10k": "📘", "10q": "📗", "8k": "⚡"}
            for src in sources:
                ftype = src["filing_type"]
                title = (
                    f"{type_icon[ftype]} **{src['ticker']}** · "
                    f"{type_label[ftype]} {src['label']} · "
                    f"score `{src['score']}`"
                )
                with st.expander(title, expanded=False):
                    s = src["summary"]
                    if ftype == "10k":
                        if s.get("top_risks"):
                            st.markdown("**Top risks**")
                            for r in s["top_risks"]:
                                st.markdown(f"- {r}")
                        if s.get("strategic_highlights"):
                            st.markdown("**Strategic highlights**")
                            for h in s["strategic_highlights"]:
                                st.markdown(f"- {h}")
                        if s.get("analyst_note"):
                            st.markdown(f"**Analyst note** — {s['analyst_note']}")
                    elif ftype == "10q":
                        c1, c2 = st.columns(2)
                        c1.metric("Revenue", s.get("revenue", "—"))
                        c2.metric("Net income", s.get("net_income", "—"))
                        if s.get("key_metrics"):
                            st.markdown("**Key metrics**")
                            for m in s["key_metrics"]:
                                st.markdown(f"- {m}")
                        if s.get("management_tone"):
                            st.markdown(f"**Management tone** — {s['management_tone']}")
                        if s.get("analyst_note"):
                            st.markdown(f"**Analyst note** — {s['analyst_note']}")
                    else:
                        c1, c2 = st.columns(2)
                        c1.metric("Event", s.get("event_type", "—"))
                        c2.metric("Materiality", s.get("materiality", "—"))
                        if s.get("headline"):
                            st.markdown(f"**Headline** — {s['headline']}")
                        if s.get("investor_impact"):
                            st.markdown(f"**Investor impact** — {s['investor_impact']}")
                    st.markdown(
                        f"[🔗 View on SEC EDGAR]({edgar_url(src['ticker'], ftype)})"
                    )

elif run_btn:
    st.warning("Please enter a question.")

# ── Footer ──────────────────────────────────────────────────────
st.divider()
st.caption("Data source: SEC EDGAR · Model: Gemini 3.1 Pro Preview via Vertex AI · Vector DB: Vertex AI Vector Search")
