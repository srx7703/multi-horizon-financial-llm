# Architecture & Design Decisions

This document explains the non-obvious engineering choices in the Multi-Horizon Financial Research Agent. The README covers *what* the system does; this doc covers *why it looks the way it does*.

---

## System overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Data                                                           │
│    SEC EDGAR (edgartools)                                       │
│      → 23 × 10-K summaries (5 tickers × 5 fiscal years — subset)│
│      → 136 × 10-Q summaries (69 tickers × last 2 quarters)      │
│      → 222 × 8-K summaries  (69 tickers × last 90 days)         │
│      → 381 docs total, all 69 tickers in RAG + distillation     │
└─────────────────────────────────────────────────────────────────┘
                           │
         ┌─────────────────┴─────────────────┐
         ▼                                   ▼
┌──────────────────────┐           ┌─────────────────────────┐
│ RAG path (Streamlit) │           │ Knowledge distillation  │
│                      │           │                         │
│ Vertex AI Vector     │           │ Gemini 3.1 Pro (Teacher)│
│ Search (3072-dim     │           │   → 1,060 QA pairs      │
│ gemini-embedding-001)│           │                         │
│       +              │           │ Gemma 2 27B (Student)   │
│ BM25 hybrid retrieval│           │   + LoRA rank=8         │
│       +              │           │   on TPU v6e-8          │
│ Gemini 3.1 Pro       │           │   (FSDPv2 / bf16)       │
│ (synthesis)          │           │                         │
└──────────────────────┘           └─────────────────────────┘
                                                │
                                                ▼
                                       BERTScore evaluation
                                       (RoBERTa-large)
```

Two pipelines share the same underlying data but answer different questions:
- **RAG path** — factual retrieval + synthesis for end-user queries.
- **Distillation path** — produces a Gemma 2 27B checkpoint specialized for SEC QA.

---

## Decision 1 — Why TPU, not Apple Silicon (MLX)?

The first attempt was MLX on a Mac Studio. Training a 27B model in 4-bit quantization is tractable on MLX, and that's exactly what `mlx-community/gemma-2-27b-it-4bit` provides.

**What broke:** the preset loads weights in fp16, not bf16. Gemma 2's attention logits exceed fp16 range (`±65504`) at sequence lengths >~200 tokens — producing NaN activations before the first LoRA step. Running a 27B model in fp16 is a latent bug, not a performance choice.

**Alternatives considered:**
- **bf16 on MLX** — MLX supports bf16, but the community-quantized checkpoint doesn't ship bf16 weights, and requantizing from the HF weights on a 128GB Mac Studio runs out of unified memory.
- **fp8 / int4 LoRA** — would need bitsandbytes kernels, which don't run on MLX.
- **Rent a single A100** — $1.50/hr for 80GB is fine for inference but not training: 27B + optimizer state + activations needs closer to 200GB.

**What we picked:** **TPU v6e-8 (Trillium)** on GCP.
- 8 chips × 32GB HBM = 256GB total — enough for full-precision bf16 training.
- bf16 is native and matches Gemma 2's original training precision.
- Spot pricing is ~$4/hr — end-to-end Phase 1 training + eval cost <$20.
- Google job applications read a "TPU + bf16 + SPMD" narrative more favorably than "local MLX" for 27B-scale work.

The MLX path wasn't wasted — it shaped the data pipeline (the `finetune_data_v2/` JSONL format was designed for MLX, then reused unchanged for HF PEFT).

---

## Decision 2 — Why LoRA rank 8?

Three rank options were considered: 4, 8, 16.

| Rank | Adapter size | Trainable params | Rationale |
|------|-------------:|-----------------:|-----------|
| 4    |  ~115 MB     |   ~27M (0.1%)    | Too narrow — leaves headroom on the table |
| **8** | **~228 MB** | **~54M (0.2%)** | **Picked** — matches the literature's "sweet spot" for 10B+ models |
| 16   |  ~455 MB     |   ~108M (0.4%)   | Marginal gain on ~1K examples; costs 2× bandwidth during save/load |

At our dataset size (1,060 training examples, 160 used for the Phase 1 run to keep TPU hours bounded), rank 8 saturates the information the QA data can teach. Rank 16 would only help if we had ≥10× the data.

LoRA is applied to **all attention + MLP projections**:
`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.

Training only the attention projections (the default in many LoRA tutorials) underfits Gemma 2's GLU MLPs, which carry a lot of representational capacity. Including the gate/up/down projections is what lets the adapter actually move style and domain vocabulary.

---

## Decision 3 — Why FSDPv2, not tensor-parallel, not pipeline-parallel?

Gemma 2 27B in bf16 is:
- 27B params × 2 bytes = **54 GB just for weights**
- + optimizer state (AdamW: 2× weights = 108 GB for momentum + variance)
- + activations (batch × seq × hidden × layers)

That's comfortably more than a single chip's 32 GB. We need sharding. Three candidates:

| Strategy | Pro | Con |
|---|---|---|
| **Tensor-parallel** (column/row split of matmuls) | Lowest communication overhead in steady state | Requires manual per-layer shard decisions; regex on layer names breaks when upstream renames |
| **Pipeline-parallel** | Good when # devices > model depth | 27B has 46 transformer blocks; 8-way pipeline leaves stages idle (bubble) and requires micro-batching |
| **FSDP** (shard params along axis 0, all-gather on use) | Framework-automatic; no layer-by-layer config; "data-parallel with sharding" is the mental model | Every forward/backward pays an all-gather; communication-bound at small batch |

**Picked:** FSDPv2 via PyTorch/XLA SPMD. Mesh is `(8, 1)` with axes `("fsdp", "tensor")` — 8-way FSDP, no tensor-parallel. One-dimensional parallelism keeps debugging tractable; at 27B the sharding savings dominate over communication cost as long as batch × seq × hidden fills the gather windows.

```python
mesh = xs.Mesh(np.arange(8), (8, 1), ("fsdp", "tensor"))
for name, param in model.named_parameters():
    if param.ndim == 2 and param.shape[0] % 8 == 0:
        xs.mark_sharding(param, mesh, ("fsdp", None))
```

One sharp edge: `mark_sharding` requires **XLA tensors, not CPU tensors**. The correct order is `.to(device)` → iterate params → `mark_sharding`. The reverse order errors with `RuntimeError: Input tensor is not an XLA tensor`.

---

## Decision 4 — Why HuggingFace PEFT, not keras-hub?

The first TPU training attempt used `keras-hub` with Gemma 2's JAX preset. It ran, logged a falling loss, and saved a LoRA adapter — but inspection of the saved weights showed **`lora_B` was identically zero across every layer**. The forward pass composed `x + lora_B @ lora_A @ x`, which with `lora_B = 0` is just the base model. We had trained nothing.

The root cause was a subtle init-order bug in how keras-hub's LoRA wrapper interacted with the SPMD mesh — `lora_B` was initialized after `mark_sharding`, silently zeroed during the first shard sync.

**Fix:** switch to HuggingFace `peft` library, which is the de-facto standard LoRA implementation. PyTorch/XLA's SPMD hooks into `torch.nn` the same way it hooks into eager PyTorch, so `peft.get_peft_model(model, LoraConfig(...))` works unchanged on TPU.

This also picks up PEFT's well-tested save/load, adapter merging, and target-module regex — all of which would have been reinvented against keras-hub.

**Lesson:** for LoRA on a new hardware stack, verify `lora_B.abs().max() > 0` after step 1 before trusting the loss curve.

---

## Decision 5 — Why `bs=4, grad_accum=2, seq_len=512`?

These three numbers are entangled — changing one affects the others' feasibility.

**Sequence length 512.** Our QA pairs are short: question ~50 tokens, answer 200–400 tokens. `seq=512` covers the 95th percentile cleanly. `seq=1024` doubled step time with negligible reduction in truncation (we measured 0.3% of examples truncated at 512 vs 0.1% at 1024).

**Batch size 4.** This is constrained by:
- **Sharding rule**: batch must be divisible by the FSDP axis size. With 8 chips and `axis 0 = "fsdp"`, valid batches are `{1, 2, 4, 8, 16, ...}`.
- **HBM**: bs=8 without gradient checkpointing OOMed at 34.3 GB / 31.25 GB limit. bs=4 fits at ~22 GB.
- **Throughput**: bs=2 with gradient checkpointing hit ~216 s/step (batch too small, activations recomputed in backward). bs=4 without GC is ~7.4 s/step post-compile — ~30× faster.

**Gradient accumulation 2.** Effective batch = 4 × 2 = 8, matching the chip count. This gives one optimizer update per full mesh-worth of data without blowing up HBM.

Gradient checkpointing is a common reflex for large-model LoRA, but it's a Pareto-bad choice here: bs=4 without GC has both more HBM headroom (4× less than needed) *and* faster steps (no recompute). GC only wins when activations genuinely dominate HBM — which they don't when params are FSDP-sharded across 8 chips.

---

## Decision 6 — Why a manual XLA decode loop, not `model.generate()`?

The first inference attempt called `model.generate(..., max_new_tokens=256)` on the TPU. It hung for 20+ minutes and never produced a token.

**Diagnosis:** HuggingFace's `.generate()` runs the autoregressive decode inside a `while` loop that issues new ops to the XLA graph without calling `mark_step`. XLA's lazy-tensor model accumulates all 256 decode steps into **one monolithic graph**, then tries to compile it. For a 27B model with KV-cache, this graph is enormous — compilation doesn't finish.

**Fix:** write a manual decode loop that calls `xm.mark_step()` **once per generated token**. Each forward pass has fixed shapes (`HybridCache` pre-allocates), so XLA compiles one forward graph, caches it, and reuses it for every subsequent token.

```python
for i in range(max_new_tokens - 1):
    pos = prompt_len + i
    out = model(
        input_ids=next_tok,
        position_ids=torch.tensor([[pos]], device=device),
        cache_position=torch.tensor([pos], device=device),
        past_key_values=past,
        use_cache=True,
    )
    past = out.past_key_values
    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    xm.mark_step()
```

Two non-obvious details:
1. **`HybridCache` needs explicit `cache_position` and `position_ids`.** Without them, the cache writes land in the wrong slots and greedy decoding loops on `"on the on the on the..."`. Gemma 2 uses `HybridCache` by default (mixing sliding-window and full attention layers), which pre-allocates a fixed-size cache — whoever calls the model must tell it which slot to write.
2. **`torch.utils.checkpoint` does `getattr(torch, 'xla')`.** If the script ever touches gradient checkpointing, paste `if not hasattr(torch, "xla"): torch.xla = torch_xla` right after `import torch_xla`. Otherwise `gradient_checkpointing_enable()` errors with `module 'torch' has no attribute 'xla'`.

---

## Decision 7 — Why migrate to Gemma 4

Phase 2 rebuilt the same pipeline on Gemma 4 31B. Three questions drove the choice.

**Why touch the base model at all?** A LoRA adapter is only as good as what the base can express. The Phase 1 eval (16/20 wins, +3.50% F1) was strong but not saturated — the base refused some items entirely with the "I don't have SEC data" hedge, and the adapter couldn't always overcome that. Gemma 4 is a newer pretrain on a larger / fresher corpus; if the base's refusal rate drops, the adapter has more headroom to improve.

**Why Gemma 4 31B specifically, not 27B or E4B?**

| Option | Why not |
|---|---|
| Gemma 4 27B | Doesn't exist as a dense checkpoint in the 4.x release lineup — there's a 31B dense and a 17B/4B MoE, but no 27B. |
| Gemma 4 E4B (MoE, ~4B active) | Much smaller; not a fair "same pipeline on newer base" test. Also the MoE routing would interact unpredictably with FSDPv2 all-gather. |
| **Gemma 4 31B (dense)** | **Picked.** Closest apples-to-apples swap with Gemma 2 27B, keeps the FSDPv2 sharding rule trivially valid (all hidden dims divisible by 8), and shares training precision (bf16). |

**What actually changed between the two train scripts?** Very little, by design — portability was part of the point.

| Component | Gemma 2 27B (Phase 1) | Gemma 4 31B (Phase 2) |
|---|---|---|
| `MODEL_ID` | `google/gemma-2-27b-it` | `google/gemma-4-31B-it` |
| Chat formatting | Hand-rolled `<start_of_turn>…<end_of_turn>` | `tokenizer.apply_chat_template(...)` — template lives in the tokenizer |
| `transformers` version | 4.45.2 | **≥ 5.6.2** (earlier raises `KeyError: 'gemma4'` on config load) |
| Model class | `AutoModelForCausalLM` → `Gemma2ForCausalLM` | `AutoModelForCausalLM` → `Gemma4ForConditionalGeneration` (multimodal root; we accept the frozen ~1 GB vision + audio towers rather than hand-remap state dict keys — `Gemma4ForCausalLM` alone silently loads zero weights because its expected prefix is `model.*` without the `language_model` indirection) |
| LoRA `target_modules` | Plain substring match on `q_proj` etc. | **Regex-scoped** to `.language_model.layers.\d+.(…)_proj$` — unscoped patterns also hit the vision tower's `Gemma4ClippableLinear` (1152×1152), which PEFT doesn't know how to adapt |
| Sharding rule | `axis-0 divisible by 8` | Unchanged — Gemma 4's hidden dim (5376) and intermediate dim (21504) are both multiples of 8 |
| LoRA rank / alpha / batch / optimizer | r=8, α=16, bs=4 × GA=2, AdamW lr=1e-4 | Unchanged |

The result: **+5.76% F1 on Gemma 4 vs +3.50% on Gemma 2, wins 20/20 vs 16/20, *t* = 10.42 vs 3.64**. The same adapter recipe generalizes and the effect gets larger — not smaller — on a stronger base.

**What broke during the migration, and how we fixed it.** Neither issue was about training; both were inference-side XLA pitfalls that didn't surface in Phase 1 because Gemma 2's `HybridCache` happened to work around them.

1. **`DynamicCache` + XLA = unbounded recompile.** The HuggingFace default cache grows per decode step, so each forward pass has a slightly different tensor shape. XLA recompiles on shape changes — and at 31B each recompile is minutes. The process appears to hang. Fix: pre-allocate a `StaticCache` once, left-pad every prompt to `MAX_PROMPT = 256` so the prefill shape is also fixed, and write via `cache_position` instead of appending. One compile per (prefill, decode-step) graph, reused forever after.

2. **`StaticCache.__init__` bug in `transformers` 5.6.2.** `Gemma4TextConfig` has `num_kv_shared_layers = 0`. The cache constructor does:
   ```python
   if hasattr(config, "num_kv_shared_layers"):
       layer_types = layer_types[: -config.num_kv_shared_layers]
   ```
   Python slice gotcha: `-0 == 0`, so `layer_types[:0]` = `[]`. The cache comes up with zero per-layer entries and the first decode step throws `IndexError`. We route around it in `build_static_cache()` by constructing `StaticSlidingWindowLayer` / `StaticLayer` objects directly and calling `Cache.__init__(sc, layers=layers)` — skipping the buggy `__init__` entirely. See [`generate_tpu_gemma4.py:40-61`](generate_tpu_gemma4.py).

A reasonable alternative would have been to pin `transformers` to whichever minor release has the fix, but as of this work (2026-04) that release didn't exist yet and the workaround is four lines.

---

## Evaluation methodology

- **Metric:** BERTScore F1 with `roberta-large`. Chosen over ROUGE/BLEU because SEC QA answers are paraphrase-heavy — the same correct fact can be phrased many ways, and n-gram overlap metrics punish that.
- **Test set:** 20 examples held out from `valid.jsonl` (disjoint from the 160 training examples). The **same 20 items** are scored across all four model variants so paired t-tests are valid.
- **Comparison:** 4-way — base Gemma 2 27B / Gemma 2 27B + v2 LoRA / base Gemma 4 31B / Gemma 4 31B + v2 LoRA.
- **Decoding:** greedy (argmax), same prompt template across conditions. `max_new_tokens = 128` for Phase 1, `256` for Phase 2 (Gemma 4's responses are longer before hitting `<turn|>`).

Results are reported in the README and in `evaluation_results_v2.json` (Phase 1) / `evaluation_results_phase2.json` (Phase 2 4-way).

**What this eval catches:** whether the LoRA shifted the model's style and vocabulary toward SEC-analyst responses, and whether that shift transfers across base-model generations.
**What it doesn't catch:** factual accuracy against real SEC filings (the QA pairs are themselves Gemini-generated from our summaries — circular). A future evaluation should use human-labeled QA from actual 10-K passages.

---

## What's deliberately *not* here

- **No inference endpoint.** The adapter is saved on TPU VM disk. Live Streamlit inference against the fine-tuned model would require deploying the adapter behind a Vertex AI endpoint; that's Phase 2 scope, not Phase 1.
- **No distributed training across multiple TPU slices.** A single v6e-8 slice is enough for 27B with FSDPv2; scaling to v6e-64 would need pipeline-parallel and is out of scope.
- **No quantized inference.** int8/int4 would halve memory but isn't needed when we have 8 chips' worth of HBM at inference time.
- **No dataset expansion beyond what's committed.** 1,060 training pairs from 381 summaries is enough to show a BERTScore delta; more data is a linear-return investment.
