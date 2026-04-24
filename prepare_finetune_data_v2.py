"""
v2 训练数据生成：把 10-K + 10-Q + 8-K 三类 summary 转成 Gemma 2 27B 微调 QA 对。

- 10-K → 5 条/份（长期风险、战略、MDA）                → ~115 条
- 10-Q → 4 条/份（季度 YoY、QoQ、管理层语调）          → ~540 条
- 8-K  → 3 条/份（事件影响、materiality、投资者含义）  → ~660 条

总计预估 ~1300 条（保守估计，若模型返回足量 → 可达 ~2000）。
输出 MLX / HF messages 格式，80/20 split。

Resumable: 每份 summary 生成的 QA 写到 cache file，重跑跳过。

Usage:
    python3 prepare_finetune_data_v2.py                   # 全跑
    python3 prepare_finetune_data_v2.py --type 10Q        # 只跑 10-Q
    python3 prepare_finetune_data_v2.py --workers 4
"""

import os
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types

# ── Config ──────────────────────────────────────────────────────
PROJECT_ID = "project-1faae058-abd0-4492-82f"
LOCATION   = "global"
MODEL      = "gemini-3.1-pro-preview"

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DIR_10K       = os.path.join(BASE_DIR, "summaries")
DIR_10Q       = os.path.join(BASE_DIR, "summaries_10q")
DIR_8K        = os.path.join(BASE_DIR, "summaries_8k")
CACHE_DIR     = os.path.join(BASE_DIR, "finetune_data_v2", "qa_cache")
OUT_DIR       = os.path.join(BASE_DIR, "finetune_data_v2")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR,  exist_ok=True)

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

SYSTEM = ("You are a senior Wall Street equity analyst with deep expertise in SEC filings "
          "and financial statement analysis. Provide precise, data-driven insights.")

# ── Prompts per filing type ─────────────────────────────────────
PROMPT_10K = """Based on this company ANNUAL 10-K summary, generate 5 financial analysis Q&A pairs.

Focus on: long-term risks, strategic positioning, MDA interpretation, year-over-year trajectory.

Company: {ticker}, Fiscal Year: {year}
Risks: {risks}
Highlights: {highlights}
MDA: {mda}
Analyst Note: {note}

Output a JSON array of 5 objects with "question" and "answer" keys.
Questions: specific, analytical, reference {ticker} FY{year} explicitly.
Answers: 2-3 sentences, cite specific figures/phrases from the summary. Do NOT hallucinate numbers not present.
Output ONLY valid JSON array."""

PROMPT_10Q = """Based on this QUARTERLY 10-Q summary, generate 4 financial analysis Q&A pairs.

Focus on: YoY/QoQ revenue + earnings changes, new risks vs prior quarter, management tone signals, quarter-specific takeaways.

Company: {ticker}, Period: {period}
Revenue: {revenue}
Net Income: {net_income}
Key Metrics: {metrics}
QoQ Changes: {qoq}
New Risks: {risks}
Management Tone: {tone}
Analyst Note: {note}

Output a JSON array of 4 objects with "question" and "answer" keys.
Questions: reference {ticker} {period} (e.g. "In {ticker}'s {period}..."). Be specific.
Answers: 2-3 sentences citing exact figures from the summary.
Output ONLY valid JSON array."""

PROMPT_8K = """Based on this 8-K current-event filing summary, generate 3 analyst Q&A pairs.

Focus on: what happened, materiality to the investment thesis, forward-looking implications for investors.

Company: {ticker}, Filing Date: {date}
Event Type: {event_type}
Headline: {headline}
Materiality: {materiality}
Investor Impact: {impact}
Key Figures/Names: {figures}

Output a JSON array of 3 objects with "question" and "answer" keys.
Questions: reference the event specifically (e.g. "What did {ticker}'s {date} 8-K disclose about...").
Answers: 2-3 sentences, cite headline + key figures. Explain materiality and investor takeaway.
Output ONLY valid JSON array."""

# ── Generation ──────────────────────────────────────────────────
def generate_qa(prompt, retries=2):
    for attempt in range(retries + 1):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    response_mime_type="application/json",
                ),
            )
            data = json.loads(resp.text)
            if isinstance(data, list):
                return [p for p in data if isinstance(p, dict) and "question" in p and "answer" in p]
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
    return []

def process_10k(fname):
    cache = os.path.join(CACHE_DIR, f"10k_{fname}")
    if os.path.exists(cache):
        return json.load(open(cache))
    s = json.load(open(os.path.join(DIR_10K, fname)))
    prompt = PROMPT_10K.format(
        ticker=s.get("ticker", ""),
        year=s.get("fiscal_year", ""),
        risks="; ".join(s.get("top_risks", [])),
        highlights="; ".join(s.get("strategic_highlights", [])),
        mda=s.get("mda_summary", ""),
        note=s.get("analyst_note", ""),
    )
    pairs = generate_qa(prompt)
    json.dump(pairs, open(cache, "w"))
    return pairs

def process_10q(fname):
    cache = os.path.join(CACHE_DIR, f"10q_{fname}")
    if os.path.exists(cache):
        return json.load(open(cache))
    s = json.load(open(os.path.join(DIR_10Q, fname)))
    prompt = PROMPT_10Q.format(
        ticker=s.get("ticker", ""),
        period=s.get("fiscal_period", s.get("period_end", "")),
        revenue=s.get("revenue", ""),
        net_income=s.get("net_income", ""),
        metrics="; ".join(s.get("key_metrics", [])),
        qoq="; ".join(s.get("qoq_changes", [])),
        risks="; ".join(s.get("new_risks", [])),
        tone=s.get("management_tone", ""),
        note=s.get("analyst_note", ""),
    )
    pairs = generate_qa(prompt)
    json.dump(pairs, open(cache, "w"))
    return pairs

def process_8k(fname):
    cache = os.path.join(CACHE_DIR, f"8k_{fname}")
    if os.path.exists(cache):
        return json.load(open(cache))
    s = json.load(open(os.path.join(DIR_8K, fname)))
    prompt = PROMPT_8K.format(
        ticker=s.get("ticker", ""),
        date=s.get("filing_date", ""),
        event_type=s.get("event_type", ""),
        headline=s.get("headline", ""),
        materiality=s.get("materiality", ""),
        impact=s.get("investor_impact", ""),
        figures="; ".join(map(str, s.get("key_figures", []))),
    )
    pairs = generate_qa(prompt)
    json.dump(pairs, open(cache, "w"))
    return pairs

HANDLERS = {"10K": (DIR_10K, process_10k), "10Q": (DIR_10Q, process_10q), "8K": (DIR_8K, process_8k)}

def run_type(kind, workers):
    directory, fn = HANDLERS[kind]
    if not os.path.isdir(directory):
        print(f"  [{kind}] dir not found: {directory}")
        return []
    files = sorted(f for f in os.listdir(directory) if f.endswith(".json"))
    print(f"\n▶ {kind}: {len(files)} summaries")
    all_pairs = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fn, f): f for f in files}
        for i, fut in enumerate(as_completed(futures), 1):
            fname = futures[fut]
            try:
                pairs = fut.result()
                all_pairs.extend(pairs)
                if i % 20 == 0 or i == len(files):
                    print(f"  [{i}/{len(files)}] total QA so far: {len(all_pairs)}")
            except Exception as e:
                print(f"  ❌ {fname}: {str(e)[:80]}")
    print(f"  {kind} done: {len(all_pairs)} QA pairs")
    return all_pairs

# ── Driver ──────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["10K", "10Q", "8K", "all"], default="all")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    kinds = ["10K", "10Q", "8K"] if args.type == "all" else [args.type]
    t0 = time.time()
    all_pairs = []
    for k in kinds:
        all_pairs.extend(run_type(k, args.workers))

    # Shuffle then split 80/20
    import random
    random.seed(42)
    random.shuffle(all_pairs)
    split = int(len(all_pairs) * 0.8)
    train, valid = all_pairs[:split], all_pairs[split:]

    def to_record(p):
        return {"messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": p["question"]},
            {"role": "assistant", "content": p["answer"]},
        ]}

    with open(os.path.join(OUT_DIR, "train.jsonl"), "w") as f:
        for p in train:
            f.write(json.dumps(to_record(p)) + "\n")
    with open(os.path.join(OUT_DIR, "valid.jsonl"), "w") as f:
        for p in valid:
            f.write(json.dumps(to_record(p)) + "\n")

    print(f"\n✅ Total: {len(all_pairs)} QA → train {len(train)} / valid {len(valid)}")
    print(f"   {OUT_DIR}/train.jsonl")
    print(f"   {OUT_DIR}/valid.jsonl")
    print(f"⏱  {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
