# Multi-Horizon Financial Research Agent

> Domain-specialized Gemma 2 27B for SEC filings, fine-tuned on TPU v6e-8 with PyTorch/XLA FSDPv2 — plus a Vertex AI Vector Search RAG demo for interactive queries.

![TPU](https://img.shields.io/badge/TPU-v6e--8%20Trillium-4285F4?logo=google-cloud)
![Gemma](https://img.shields.io/badge/Gemma%202%2027B-LoRA%20fine--tuned-EA4335?logo=google)
![PyTorch XLA](https://img.shields.io/badge/PyTorch-XLA%20SPMD%20FSDPv2-EE4C2C?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-RAG%20demo-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/license-MIT-green)

---

## TL;DR

- **What:** a Gemma 2 27B LoRA adapter specialized for SEC financial QA, trained on 1,060 knowledge-distilled QA pairs from 381 SEC filing summaries (10-K, 10-Q, 8-K).
- **Where it runs:** TPU v6e-8 (Trillium) via HuggingFace PEFT + PyTorch/XLA with SPMD FSDPv2 param-sharding in bf16.
- **Headline result:** **BERTScore F1 0.8078 → 0.8361 (+3.50% relative, +0.0284 absolute) on 20 held-out SEC QA items**, paired *t* = 3.64 (p < 0.01), 95% CI on the delta = [+0.012, +0.045], v2 beats base on 16/20 items. See [§ Model evaluation](#model-evaluation).
- **Separate RAG demo:** Streamlit app over Vertex AI Vector Search + Gemini 3.1 Pro, covering 5 companies × 10-K for interactive exploration. See [§ Streamlit demo](#streamlit-demo).

Full trade-off discussion in [ARCHITECTURE.md](ARCHITECTURE.md).

---

## What this is (and isn't)

**This is two things, sharing a data pipeline:**

1. **Knowledge-distillation pipeline** — 69 S&P 500 companies' SEC filings (10-K + 10-Q + 8-K) are summarized by Gemini 3.1 Pro, turned into 1,060 analyst-style QA pairs, and used to fine-tune Gemma 2 27B with LoRA on TPU. Evaluated by BERTScore.
2. **RAG demo** — a smaller Streamlit app that runs hybrid retrieval (dense + BM25) over 23 × 10-K summaries for 5 companies in Vertex AI Vector Search, then synthesizes answers with Gemini 3.1 Pro.

The RAG demo exists to make the data pipeline tangible — you can ask questions and see the retrieval + generation flow. The Gemma fine-tune is the research artifact; it is not served from the Streamlit app (the adapter lives on the TPU VM; live inference from a laptop isn't practical for 27B).

---

## Pipeline at a glance

```
SEC EDGAR API (edgartools)
   ├─ 10-K summaries — 23 (5 companies × 5 years)   → Vertex AI Vector Search → RAG demo
   ├─ 10-Q summaries — 136 (69 companies × 2 quarters)
   └─ 8-K  summaries — 222 (69 companies × 90 days)
         │
         ▼
Gemini 3.1 Pro (Teacher)
   → 1,060 analyst-style QA pairs (finetune_data_v2/train.jsonl)
         │
         ▼
Gemma 2 27B (Student) + LoRA rank=8
   FSDPv2 param-sharding across 8 TPU v6e chips (bf16)
   → gemma27b_financial_adapter_hf/  (228 MB)
         │
         ▼
BERTScore eval (RoBERTa-large)  vs base Gemma 2 27B
   → evaluation_results_v2.json
```

---

## Model evaluation

Comparison on 20 held-out SEC QA items drawn from `finetune_data_v2/valid.jsonl` (no overlap with the 160 training examples used for the Phase 1 run).

| Model | BERTScore F1 | BERTScore P | BERTScore R |
|---|---:|---:|---:|
| Base Gemma 2 27B (no adapter) | 0.8078 | 0.7992 | 0.8167 |
| **Gemma 2 27B + LoRA (SEC data)** | **0.8361** | **0.8297** | **0.8439** |
| **Delta (relative)** | **+3.50%** | **+3.82%** | **+3.33%** |

**Statistical check (paired, n=20):**

| Statistic | Value |
|---|---|
| Mean F1 delta (v2 − base) | +0.0284 |
| Standard error | 0.0078 |
| Paired *t* (df=19) | **3.64** (p < 0.01) |
| 95% CI for the delta | [+0.0120, +0.0447] |
| Items where v2 beats base | **16 / 20** |

Effect is robust at this sample size: the 95% CI doesn't cross zero, and v2 wins on 80% of items.

**Qualitative change.** The base model refuses most financial questions with *"I do not have access to real-time data, including SEC filings..."* — safety boilerplate triggered by the analyst framing. The fine-tuned model produces structured analyst responses citing specific filings (*"The 2026-03-02 8-K is highly material to investors because..."*). The BERTScore delta captures a style shift more than a factual shift — which is exactly what LoRA on distilled-QA data is designed to produce.

Full per-item predictions: [`preds/preds_base.json`](preds/preds_base.json), [`preds/preds_v2.json`](preds/preds_v2.json).
Full report: [`evaluation_results_v2.json`](evaluation_results_v2.json).

---

## Fine-tuning setup

| Component | Value |
|---|---|
| Base model | `google/gemma-2-27b-it` (bf16) |
| Hardware | TPU v6e-8 (Trillium), 256 GB HBM total |
| Framework | HuggingFace `transformers` 4.45.2 + `peft` 0.13.2 + `torch-xla` 2.5.0 |
| Parallelism | SPMD FSDPv2, mesh `(8, 1)` over `("fsdp", "tensor")` |
| LoRA | rank=8, alpha=16, dropout=0.05 |
| LoRA targets | `q/k/v/o/gate/up/down_proj` (attention + MLP) |
| Sequence length | 512 |
| Batch size | 4 × 2 grad-accum = effective 8 |
| Optimizer | AdamW, lr=1e-4, wd=0.01 |
| Epochs | 2 over 160 examples (cap; full data is 1,060) |
| Precision | bf16 (matches Gemma 2 training precision) |
| Step time (post-compile) | ~7.4 s / optimizer step |
| Trained adapter size | 228 MB |

Training loss fell from 3.21 → 1.16 over 40 optimizer steps. See [ARCHITECTURE.md](ARCHITECTURE.md) for why each number is what it is, including why `bs=4 no-GC` beat `bs=2 with gradient checkpointing` by ~30×.

---

## Streamlit demo

```bash
pip install -r requirements.txt
gcloud auth application-default login
streamlit run app.py
# → http://localhost:8501
```

The app demonstrates the RAG side of the pipeline over 5 companies (Apple, NVIDIA, Tesla, Microsoft, Rivian) × 5 years of 10-K filings:

- Query → `gemini-embedding-001` → Vertex AI Vector Search (3072-dim, 23 summaries + 69 paragraph chunks) → top-k retrieval
- Optional BM25 hybrid re-rank (`alpha` slider: 0.35 = recall-heavy, 0.6 = balanced, 0.8 = precision-heavy)
- Context passed to Gemini 3.1 Pro for Wall Street analyst-style synthesis

The fine-tuned Gemma 2 27B is **not** exposed by the app (27B bf16 can't run on a laptop; endpoint deployment is out of Phase 1 scope). Model evaluation numbers shown in the UI sidebar come from `evaluation_results_v2.json`.

Demo GIF: `docs/demo.gif` *(to be added)*.

---

## Data coverage

**For RAG (in Vertex AI Vector Search):**

| Company | Ticker | Years |
|---|---|---|
| Apple | AAPL | 2021–2025 |
| NVIDIA | NVDA | 2022–2026 |
| Tesla | TSLA | 2022–2025 |
| Microsoft | MSFT | 2021–2025 |
| Rivian | RIVN | 2021–2025 |

23 × 10-K summary chunks + 69 × paragraph chunks.

**For fine-tuning (in `finetune_data_v2/`):**

- 69 S&P 500 companies
- 3 filing types per company: 10-K (last 5 years), 10-Q (last 2 quarters), 8-K (last 90 days)
- 381 structured summaries → 1,060 teacher-generated QA pairs

The distillation corpus is 16× the size of the RAG index — scaling up the Vector Search index to 69 companies is straightforward but out of Phase 1 scope.

---

## Repo layout

```
├── train_tpu_hf_peft.py           # LoRA fine-tuning on TPU v6e-8
├── generate_tpu_hf.py             # Inference with manual XLA decode loop
├── compute_bertscore_v2.py        # Local BERTScore computation vs base
├── prepare_finetune_data_v2.py    # Teacher-generated QA from SEC summaries
├── sec_expand.py                  # SEC EDGAR fetch + Gemini summarization
├── app.py                         # Streamlit RAG demo
├── rag_with_gemma.py              # Optional Gemma-in-the-loop RAG variant
├── hybrid_retriever.py            # Dense + BM25 hybrid retrieval
├── finetune_data_v2/              # 1,060 train + 265 valid QA pairs
├── summaries/                     # 10-K JSON summaries (5 co × 5 yr)
├── summaries_10q/                 # 10-Q JSON summaries (69 co)
├── preds/                         # Held-out BERTScore predictions
├── evaluation_results_v2.json     # Full evaluation report
├── ARCHITECTURE.md                # Trade-off narrative
└── ROADMAP.md                     # Phase 1 / Phase 2 plan
```

---

## Reproducing the TPU training

This requires a v6e-8 TPU VM on GCP with 256 GB HBM; full instructions in [ARCHITECTURE.md](ARCHITECTURE.md). Sketch:

```bash
# On the TPU VM
pip install transformers==4.45.2 peft==0.13.2 torch-xla==2.5.0
scp finetune_data_v2/train.jsonl tpu:~/train.jsonl

# Training (~8 min for 40 opt-steps post-compile; cold compile adds ~6 min)
PJRT_DEVICE=TPU PYTHONUNBUFFERED=1 python3 train_tpu_hf_peft.py

# Adapter lands at ~/gemma27b_financial_adapter_hf/
# Generate and scp back for local BERTScore scoring:
PJRT_DEVICE=TPU python3 generate_tpu_hf.py --mode base --n 20 --max-new-tokens 128
PJRT_DEVICE=TPU python3 generate_tpu_hf.py --mode v2 \
    --adapter ~/gemma27b_financial_adapter_hf --n 20 --max-new-tokens 128
```

Budget: Phase 1 end-to-end (training + 2 × inference sweeps + eval) is ~90 TPU minutes at ~$4/hr spot = **<$10** on `v6e-8`.

---

## GCP resources (RAG side)

- **Project:** `project-1faae058-abd0-4492-82f`
- **Vector Search endpoint:** `2952648316438970368` (`us-central1`)
- **Deployed index:** `sec_financial_deployed`
- **Embedding model:** `gemini-embedding-001` (3072-dim)
- **Generation model:** `gemini-3.1-pro-preview` (`global` location)

---

## Roadmap

- ✅ **Phase 1** — Gemma 2 27B specialization and evaluation (this README)
- ⏳ **Phase 2** — migrate the same pipeline to Gemma 4 27B, do a 4-way BERTScore comparison (base2 / base2+v2 / base4 / base4+v2) showing pipeline portability

See [ROADMAP.md](ROADMAP.md).

---

## License

MIT. Data derived from public SEC EDGAR filings.

---

*Built on: SEC EDGAR · Gemini 3.1 Pro · Gemma 2 27B · TPU v6e-8 · PyTorch/XLA · Vertex AI · Streamlit · BERTScore*
