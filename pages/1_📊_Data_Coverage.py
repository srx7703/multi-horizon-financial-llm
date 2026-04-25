"""Data Coverage page — full ticker × filing-type breakdown."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Data Coverage", page_icon="📊", layout="wide")

ROOT = Path(__file__).parent.parent
SUMMARY_DIRS = {
    "10-K": ROOT / "summaries",
    "10-Q": ROOT / "summaries_10q",
    "8-K":  ROOT / "summaries_8k",
}


@st.cache_data
def scan_coverage():
    coverage: dict[str, dict] = {}
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
            row = coverage.setdefault(
                ticker,
                {"Ticker": ticker, "10-K": 0, "10-Q": 0, "8-K": 0},
            )
            row[ftype] += 1
    return coverage


def edgar_url(ticker: str) -> str:
    return (
        "https://www.sec.gov/cgi-bin/browse-edgar"
        f"?action=getcompany&CIK={ticker}&type=&dateb=&owner=include&count=40"
    )


coverage = scan_coverage()
df = pd.DataFrame(coverage.values())
df["Total"] = df["10-K"] + df["10-Q"] + df["8-K"]
df["EDGAR"] = df["Ticker"].apply(edgar_url)
df = df.sort_values("Total", ascending=False).reset_index(drop=True)

st.title("📊 Data Coverage")
st.caption(
    f"SEC EDGAR filings indexed in the RAG corpus — "
    f"**{len(df)} S&P 500 tickers · {df['Total'].sum()} filings** "
    f"({df['10-K'].sum()} 10-K · {df['10-Q'].sum()} 10-Q · {df['8-K'].sum()} 8-K)"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tickers", len(df))
c2.metric("10-K (annual)", int(df["10-K"].sum()))
c3.metric("10-Q (quarterly)", int(df["10-Q"].sum()))
c4.metric("8-K (event)", int(df["8-K"].sum()))

st.divider()

q = st.text_input("Filter by ticker", placeholder="e.g. AAPL, NVDA")
view = df if not q else df[df["Ticker"].str.contains(q.upper().strip())]

st.dataframe(
    view,
    column_config={
        "Ticker": st.column_config.TextColumn("Ticker", width="small"),
        "10-K": st.column_config.NumberColumn("10-K", width="small"),
        "10-Q": st.column_config.NumberColumn("10-Q", width="small"),
        "8-K":  st.column_config.NumberColumn("8-K",  width="small"),
        "Total": st.column_config.NumberColumn("Total", width="small"),
        "EDGAR": st.column_config.LinkColumn(
            "SEC EDGAR",
            display_text="View filings →",
            help="Open this ticker's full filing history on SEC EDGAR",
        ),
    },
    hide_index=True,
    use_container_width=True,
    height=min(800, 36 + 35 * len(view)),
)

st.divider()
st.caption(
    "Source: SEC EDGAR · Each filing distilled into a structured summary by Gemini 3.1 Pro · "
    "Used for both knowledge distillation training (LoRA) and RAG retrieval."
)
