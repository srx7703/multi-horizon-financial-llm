"""
SEC Expansion Pipeline: Pull 10-Q + 8-K for 69 companies, summarize via Gemini 3.1 Pro

Architecture:
- 10-Q  → 季度财务更新（近 2 季度）        → summaries_10q/
- 8-K   → 突发事件公告（近 90 天）          → summaries_8k/

Rate limits:
- SEC EDGAR: 10 req/sec max (edgartools handles)
- Gemini API: 60 RPM on paid tier (we sleep 1s between calls)

Resumable: skips files that already exist.

Usage:
    python3 sec_expand.py --filing 10Q     # only 10-Q
    python3 sec_expand.py --filing 8K      # only 8-K
    python3 sec_expand.py --filing all     # both (default)
    python3 sec_expand.py --tickers NVDA,AAPL  # subset for testing
"""

import os
import json
import time
import argparse
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from edgar import Company, set_identity
from google import genai
from google.genai import types

# ── Config ──────────────────────────────────────────────────────
PROJECT_ID  = "project-1faae058-abd0-4492-82f"
LOCATION    = "global"
MODEL       = "gemini-3.1-pro-preview"
SEC_IDENTITY = "Ruoxuan Song ruoxuan@example.com"  # SEC requires identity

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DIR_10Q       = os.path.join(BASE_DIR, "summaries_10q")
DIR_8K        = os.path.join(BASE_DIR, "summaries_8k")

TICKERS_69 = [
    "AAPL","ABBV","ABNB","AMC","AMD","AMZN","AVGO","AXP","AZO","BAC",
    "BRK-A","CAT","CB","CMCSA","CMG","COST","CRM","CVS","CVX","DAL",
    "DLTR","DVA","EA","EBAY","EFX","ENPH","ETSY","F","FDX","GILD",
    "GIS","GM","GME","GOOGL","GRMN","GS","HAS","HD","HLT","HPE",
    "HPQ","HSY","HUM","IBM","ICE","INTU","JNJ","JPM","KO","KR",
    "LLY","LULU","LVS","META","MSFT","NFLX","NKE","NVDA","PG","PLTR",
    "PTON","SBUX","SCHW","T","TSLA","UNH","V","WMT",
]

N_QUARTERS_BACK = 2     # last 2 quarters of 10-Q
DAYS_BACK_8K    = 90    # last 90 days of 8-K

os.makedirs(DIR_10Q, exist_ok=True)
os.makedirs(DIR_8K,  exist_ok=True)

set_identity(SEC_IDENTITY)
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# ── Gemini summarization prompts ────────────────────────────────
PROMPT_10Q = """You are a senior Wall Street equity research analyst reviewing a 10-Q quarterly filing.

Extract a structured JSON summary with these fields (return ONLY valid JSON, no markdown):
{
  "ticker": "<ticker>",
  "fiscal_period": "<e.g. Q3 2024>",
  "period_end": "<YYYY-MM-DD>",
  "revenue": "<revenue figure with YoY comparison>",
  "net_income": "<net income with YoY>",
  "key_metrics": ["metric 1", "metric 2", "metric 3"],
  "qoq_changes": ["what changed vs last quarter"],
  "new_risks": ["risks emerged or intensified this quarter"],
  "management_tone": "<1 sentence — bullish/cautious/neutral + why>",
  "analyst_note": "<1-2 sentence quarterly takeaway>"
}

=== 10-Q FILING ===
{filing_text}
=== END ==="""

PROMPT_8K = """You are a senior Wall Street equity research analyst reviewing an 8-K current event filing.

Extract a structured JSON summary (return ONLY valid JSON, no markdown):
{
  "ticker": "<ticker>",
  "filing_date": "<YYYY-MM-DD>",
  "event_type": "<e.g. M&A, Executive Change, Earnings, Material Agreement, Guidance Update>",
  "headline": "<1 sentence summary of the event>",
  "materiality": "<High | Medium | Low>",
  "investor_impact": "<1-2 sentence analyst take on how this affects the investment thesis>",
  "key_figures": ["specific numbers, dates, names mentioned"]
}

=== 8-K FILING ===
{filing_text}
=== END ==="""

# ── Fetch + summarize helpers ───────────────────────────────────
def safe_filename(s):
    return s.replace("/", "_").replace(" ", "_")

def summarize(text, prompt_template, max_chars=40000):
    """Call Gemini to produce structured summary. Truncate long filings."""
    text = text[:max_chars]
    prompt = prompt_template.replace("{filing_text}", text)
    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )
    return json.loads(resp.text)

def process_10q(ticker):
    results = []
    try:
        company = Company(ticker)
        filings_iter = company.get_filings(form="10-Q").latest(N_QUARTERS_BACK)
        filings = list(filings_iter) if filings_iter else []
        if not filings:
            return ticker, [], "no 10-Q filings found"

        for filing in filings:
            period = str(filing.period_of_report)  # e.g. "2024-09-28"
            out_path = os.path.join(DIR_10Q, f"{ticker}_{period}_10Q.json")
            if os.path.exists(out_path):
                results.append(f"skip {period}")
                continue

            try:
                tenq = filing.obj()
                financial_text = ""
                for section in ["management_discussion", "financial_statements", "risk_factors"]:
                    try:
                        s = getattr(tenq, section, None)
                        if s: financial_text += f"\n\n=== {section.upper()} ===\n{str(s)[:10000]}"
                    except Exception:
                        pass
                if not financial_text:
                    financial_text = filing.text()[:30000]
            except Exception:
                financial_text = filing.text()[:30000]

            summary = summarize(financial_text, PROMPT_10Q)
            summary["ticker"] = ticker
            with open(out_path, "w") as f:
                json.dump(summary, f, indent=2)
            results.append(f"ok {period}")
            time.sleep(0.8)

        return ticker, results, None
    except Exception as e:
        return ticker, results, str(e)[:200]

def process_8k(ticker):
    results = []
    try:
        company = Company(ticker)
        cutoff = datetime.now() - timedelta(days=DAYS_BACK_8K)
        filings_obj = company.get_filings(form="8-K")
        # Filter by date
        recent = [f for f in filings_obj if f.filing_date and
                  datetime.strptime(str(f.filing_date), "%Y-%m-%d") > cutoff]

        if not recent:
            return ticker, [], "no recent 8-K"

        for filing in recent[:10]:  # cap at 10 per ticker to control cost
            fdate = str(filing.filing_date)
            accession = filing.accession_number.replace("-", "")
            out_path = os.path.join(DIR_8K, f"{ticker}_{fdate}_{accession[-6:]}_8K.json")
            if os.path.exists(out_path):
                results.append(f"skip {fdate}")
                continue

            try:
                text = filing.text()[:20000]
            except Exception:
                continue

            if len(text.strip()) < 200:  # skip empty filings
                continue

            try:
                summary = summarize(text, PROMPT_8K)
                summary["ticker"] = ticker
                summary["filing_date"] = fdate
                with open(out_path, "w") as f:
                    json.dump(summary, f, indent=2)
                results.append(f"ok {fdate}")
                time.sleep(0.8)
            except Exception as e:
                results.append(f"err {fdate}: {str(e)[:80]}")

        return ticker, results, None
    except Exception as e:
        return ticker, results, str(e)[:200]

# ── Driver ──────────────────────────────────────────────────────
def run(filing_type, tickers, max_workers=3):
    fn = {"10Q": process_10q, "8K": process_8k}[filing_type]
    print(f"\n▶ {filing_type} for {len(tickers)} tickers")

    done, failed = 0, 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fn, t): t for t in tickers}
        for i, future in enumerate(as_completed(futures), 1):
            ticker, results, err = future.result()
            if err:
                failed += 1
                print(f"  [{i}/{len(tickers)}] ❌ {ticker}: {err}")
            else:
                done += 1
                print(f"  [{i}/{len(tickers)}] ✓ {ticker}: {len(results)} files — {', '.join(results[:3])}")
    print(f"\n{filing_type} summary: {done} done, {failed} failed")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filing", choices=["10Q", "8K", "all"], default="all")
    ap.add_argument("--tickers", default=None, help="comma-separated subset, e.g. NVDA,AAPL")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    tickers = args.tickers.split(",") if args.tickers else TICKERS_69

    print(f"Project: {PROJECT_ID}")
    print(f"Tickers: {len(tickers)}")
    print(f"10-Q → {DIR_10Q}")
    print(f"8-K  → {DIR_8K}")

    t0 = time.time()
    if args.filing in ["10Q", "all"]:
        run("10Q", tickers, args.workers)
    if args.filing in ["8K", "all"]:
        run("8K", tickers, args.workers)
    print(f"\n⏱  Total: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
