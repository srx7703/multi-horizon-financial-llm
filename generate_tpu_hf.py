"""
Generate predictions on TPU using HF + PEFT loaded adapter.
Produces preds_{mode}.json for ingest by compute_bertscore_v2.py.

Usage:
    PJRT_DEVICE=TPU python3 generate_tpu_hf.py --mode base
    PJRT_DEVICE=TPU python3 generate_tpu_hf.py --mode v2 --adapter /home/zczqrso/gemma27b_financial_adapter_hf
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
from peft import PeftModel

MODEL_ID  = "google/gemma-2-27b-it"
VALID     = "/home/zczqrso/valid.jsonl"
SYSTEM    = ("You are a senior Wall Street equity analyst with deep expertise in SEC filings "
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

def format_prompt(question):
    user = f"{SYSTEM}\n\n{question}"
    return (f"<bos><start_of_turn>user\n{user}<end_of_turn>\n"
            f"<start_of_turn>model\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["base", "v2"])
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    if args.mode == "v2" and not args.adapter:
        raise SystemExit("--adapter required for mode v2")

    device = xm.xla_device()
    num_devices = xr.global_runtime_device_count()
    mesh = xs.Mesh(np.arange(num_devices), (num_devices, 1), ("fsdp", "tensor"))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[{args.mode}] loading model (bf16)")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    print(f"  base model loaded in {time.time()-t0:.1f}s")

    if args.mode == "v2":
        print(f"[{args.mode}] attaching adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)

    model = model.to(device)
    for _, param in model.named_parameters():
        if param.ndim == 2 and all(d % num_devices == 0 for d in [param.shape[0]]):
            xs.mark_sharding(param, mesh, ("fsdp", None))

    model.eval()

    tests = load_tests(VALID, args.n)
    print(f"[{args.mode}] {len(tests)} tests")

    # Manual decode loop: HF's .generate() builds one lazy graph for all max_new_tokens
    # which XLA then has to compile as a monolith — times out on 27B. Instead we step
    # token-by-token with xm.mark_step per token so each forward compiles once (shape
    # is fixed by HybridCache) and then reuses the cached graph.
    eos_id = tokenizer.eos_token_id
    # Gemma 2 uses <end_of_turn> as a stop token too
    end_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")

    @torch.no_grad()
    def xla_decode(input_ids, attention_mask, max_new_tokens):
        # Prefill: positions [0..prompt_len-1]
        prompt_len = input_ids.shape[1]
        cache_position = torch.arange(prompt_len, device=device)
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cache_position=cache_position,
            use_cache=True,
        )
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = [next_tok]
        xm.mark_step()
        # Decode: each step writes 1 new slot at position prompt_len + i
        for i in range(max_new_tokens - 1):
            pos = prompt_len + i
            cache_position = torch.tensor([pos], device=device)
            position_ids = torch.tensor([[pos]], device=device)
            out = model(
                input_ids=next_tok,
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_tok)
            xm.mark_step()
            tok_id = next_tok.item()
            if tok_id == eos_id or tok_id == end_turn_id:
                break
        return torch.cat(generated, dim=1)

    results = []
    for i, item in enumerate(tests, 1):
        prompt = format_prompt(item["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        t_start = time.time()
        gen_ids = xla_decode(inputs["input_ids"], inputs["attention_mask"], args.max_new_tokens)
        pred = tokenizer.decode(gen_ids[0], skip_special_tokens=False)
        for marker in ["<end_of_turn>", "<start_of_turn>", "<eos>"]:
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

    out = f"/home/zczqrso/preds_{args.mode}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ saved → {out}")

if __name__ == "__main__":
    main()
