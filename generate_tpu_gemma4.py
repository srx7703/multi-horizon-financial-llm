"""
Generate predictions on TPU from Gemma 4 31B (base or with LoRA adapter).

Forked from generate_tpu_hf.py. Changes vs Gemma 2 27B version:
  - MODEL_ID: google/gemma-4-31B-it.
  - Chat template: use tok.apply_chat_template(add_generation_prompt=True)
    instead of hand-rolled `<bos><start_of_turn>…` (Gemma 4 uses `<|turn>…<turn|>`).
  - Stop tokens: `<turn|>` (token 106) replaces Gemma 2's `<end_of_turn>`.
  - Output files: preds_gemma4_{mode}.json to avoid collision with Phase 1 preds.
  - StaticCache + left-pad prompts to MAX_PROMPT so every decode step has the
    same tensor shapes (see `build_static_cache`). Without this the HF default
    DynamicCache grows per token, which causes XLA to recompile every decode
    iteration — in practice the process hangs indefinitely on 31B.

Usage:
    PJRT_DEVICE=TPU python3 generate_tpu_gemma4.py --mode base
    PJRT_DEVICE=TPU python3 generate_tpu_gemma4.py --mode v2g4 \\
        --adapter /home/zczqrso/gemma4_31b_financial_adapter_hf
"""
import os
os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("XLA_USE_SPMD", "1")

import argparse
import json
import time

import numpy as np
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import (
    StaticCache, StaticSlidingWindowLayer, StaticLayer, Cache,
)
from peft import PeftModel


def build_static_cache(model_config, max_cache_len):
    """
    Construct a properly-initialized StaticCache, routing around the
    transformers 5.6.2 bug where `num_kv_shared_layers == 0` triggers
    `layer_types[:-0]` → `[]` (empty slice instead of full list). We build
    the per-layer cache objects directly and graft them onto a bare Cache.
    """
    cfg = model_config.get_text_config(decoder=True)
    layer_types = cfg.layer_types or (
        ["sliding_attention"] * cfg.num_hidden_layers
    )
    layers = []
    for lt in layer_types:
        if lt == "sliding_attention":
            layers.append(StaticSlidingWindowLayer(
                max_cache_len=max_cache_len, sliding_window=cfg.sliding_window,
            ))
        else:
            layers.append(StaticLayer(max_cache_len=max_cache_len))
    sc = StaticCache.__new__(StaticCache)
    Cache.__init__(sc, layers=layers)
    return sc

MODEL_ID = "google/gemma-4-31B-it"
VALID    = "/home/zczqrso/valid.jsonl"
SYSTEM   = ("You are a senior Wall Street equity analyst with deep expertise in SEC filings "
            "and financial statement analysis. Provide precise, data-driven insights.")

def load_tests(path, n):
    items = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            msgs = r.get("messages", [])
            q = next((m["content"] for m in msgs if m["role"] == "user"), None)
            a = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
            if q and a:
                items.append({"question": q, "reference": a})
            if len(items) >= n:
                break
    return items

def format_prompt(tokenizer, question):
    msgs = [{"role": "user", "content": f"{SYSTEM}\n\n{question}"}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["base", "v2g4"])
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    if args.mode == "v2g4" and not args.adapter:
        raise SystemExit("--adapter required for mode v2g4")

    device = xm.xla_device()
    num_devices = xr.global_runtime_device_count()
    mesh = xs.Mesh(np.arange(num_devices), (num_devices, 1), ("fsdp", "tensor"))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Use AutoModelForCausalLM -> Gemma4ForConditionalGeneration so keys
    # match (see train script for why text-only Gemma4ForCausalLM can't
    # load this repo's state dict directly).
    print(f"[{args.mode}] loading model (bf16)")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    print(f"  base model loaded in {time.time()-t0:.1f}s")

    if args.mode == "v2g4":
        print(f"[{args.mode}] attaching adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)

    model = model.to(device)
    for _, param in model.named_parameters():
        if param.ndim == 2 and all(d % num_devices == 0 for d in [param.shape[0]]):
            xs.mark_sharding(param, mesh, ("fsdp", None))

    model.eval()

    tests = load_tests(VALID, args.n)
    print(f"[{args.mode}] {len(tests)} tests")

    eos_id = tokenizer.eos_token_id
    # Gemma 4 uses `<turn|>` as the per-turn terminator.
    end_turn_id = tokenizer.convert_tokens_to_ids("<turn|>")

    # Fixed-size KV cache so every decode step has static shapes — a DynamicCache
    # grows per token and triggers an XLA recompile each iteration, which hangs
    # indefinitely on 31B. StaticCache is pre-allocated to MAX_CACHE_LEN and the
    # model writes into it via `cache_position`. The per-layer tensors are
    # allocated lazily inside the model's first forward pass (device + dtype
    # inferred from the actual key/value states), so we don't need to `.to(device)`
    # here.
    MAX_PROMPT = 256
    MAX_CACHE_LEN = MAX_PROMPT + args.max_new_tokens
    static_cache = build_static_cache(model.config, MAX_CACHE_LEN)

    @torch.no_grad()
    def xla_decode(input_ids, attention_mask, max_new_tokens):
        # Left-pad prompt to MAX_PROMPT so the prefill graph is also static.
        prompt_len = input_ids.shape[1]
        if prompt_len > MAX_PROMPT:
            input_ids = input_ids[:, -MAX_PROMPT:]
            attention_mask = attention_mask[:, -MAX_PROMPT:]
            prompt_len = MAX_PROMPT
        pad_len = MAX_PROMPT - prompt_len
        if pad_len > 0:
            pad = torch.full((1, pad_len), tokenizer.pad_token_id,
                             dtype=input_ids.dtype, device=input_ids.device)
            input_ids = torch.cat([pad, input_ids], dim=1)
            zeros = torch.zeros((1, pad_len), dtype=attention_mask.dtype,
                                device=attention_mask.device)
            attention_mask = torch.cat([zeros, attention_mask], dim=1)

        static_cache.reset()
        cache_position = torch.arange(MAX_PROMPT, device=device)
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=static_cache,
            use_cache=True,
        )
        # Real last-token index = MAX_PROMPT - 1 (because we right-padded-then-
        # left-padded, so prompt ends at MAX_PROMPT-1).
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = [next_tok]
        xm.mark_step()
        for i in range(max_new_tokens - 1):
            pos = MAX_PROMPT + i
            cache_position = torch.tensor([pos], device=device)
            position_ids = torch.tensor([[pos]], device=device)
            out = model(
                input_ids=next_tok,
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_values=static_cache,
                use_cache=True,
            )
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_tok)
            xm.mark_step()
            tok_id = next_tok.item()
            if tok_id == eos_id or tok_id == end_turn_id:
                break
        return torch.cat(generated, dim=1)

    results = []
    for i, item in enumerate(tests, 1):
        prompt = format_prompt(tokenizer, item["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        t_start = time.time()
        gen_ids = xla_decode(inputs["input_ids"], inputs["attention_mask"], args.max_new_tokens)
        pred = tokenizer.decode(gen_ids[0], skip_special_tokens=False)
        for marker in ["<turn|>", "<|turn>", "<eos>", "<end_of_turn>", "<start_of_turn>"]:
            if marker in pred:
                pred = pred.split(marker)[0]
        pred = pred.strip()
        dt = time.time() - t_start
        results.append({
            "question": item["question"],
            "reference": item["reference"],
            "prediction": pred,
        })
        print(f"  [{i}/{len(tests)}] {dt:.1f}s — {pred[:80]}", flush=True)

    out = f"/home/zczqrso/preds_gemma4_{args.mode}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ saved → {out}")

if __name__ == "__main__":
    main()
