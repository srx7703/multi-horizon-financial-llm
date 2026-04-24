"""
Gemma 2 27B LoRA fine-tuning on TPU v6e-8 via HuggingFace + PEFT + PyTorch/XLA.

Replaces the earlier keras_hub + bf16 + ModelParallel path which had a bug
where lora_B stayed zero (no LoRA effect). HF PEFT's LoRA implementation is
framework-standard and well tested on TPU.

Sharding: FSDPv2 via SPMD auto-sharding across 8 TPU chips.
Precision: bfloat16 (native for Gemma 2, matches original training).
LoRA targets: all attention + MLP projection linears (matches Gemma's
    paper-standard fine-tuning surface).

Usage on TPU VM:
    PJRT_DEVICE=TPU python3 train_tpu_hf_peft.py
"""
import os
os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("XLA_USE_SPMD", "1")
# Persist compiled XLA graphs across runs so kills+restarts don't recompile.
os.environ.setdefault("XLA_PERSISTENT_CACHE_PATH", "/home/zczqrso/.cache/xla_cache")
os.makedirs(os.environ["XLA_PERSISTENT_CACHE_PATH"], exist_ok=True)

import json
import sys
import time

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr

# Patch: make gradient_checkpointing's _get_device_module('xla') work.
# PyTorch's checkpoint.py calls getattr(torch, 'xla'), but torch_xla
# isn't automatically exposed as torch.xla.
if not hasattr(torch, "xla"):
    torch.xla = torch_xla

import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader

MODEL_ID   = "google/gemma-2-27b-it"
TRAIN_FILE = "/home/zczqrso/train.jsonl"
OUT_DIR    = "/home/zczqrso/gemma27b_financial_adapter_hf"
MAX_LEN    = 512    # halved from 1024 for 2x+ speedup; 512 tokens fits most QA cleanly
BATCH_SIZE = 4      # bs=8 OOMed without GC (34G>31G); bs=4 halves activations
EPOCHS     = 2
LR         = 1e-4   # LoRA-typical
LORA_RANK  = 8
GRAD_ACCUM = 2      # effective batch = 8, gives more opt steps on the small cap
LOG_EVERY  = 1      # log every optimizer step
MAX_EXAMPLES = 160  # cap dataset for viable demo runtime

# ── Chat template helper (Gemma 2 doesn't support system role) ─────────
def messages_to_text(msgs, tokenizer):
    system = next((m["content"] for m in msgs if m["role"] == "system"), "")
    user   = next((m["content"] for m in msgs if m["role"] == "user"), "")
    asst   = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
    user_merged = (system + "\n\n" + user) if system else user
    # Match Gemma's exact template: <bos><start_of_turn>user ... <end_of_turn>
    text = (
        f"<bos><start_of_turn>user\n{user_merged}<end_of_turn>\n"
        f"<start_of_turn>model\n{asst}<end_of_turn>\n"
    )
    return text

class JsonlQADataset(Dataset):
    def __init__(self, path, tokenizer, max_len, max_examples=None):
        self.examples = []
        with open(path) as f:
            for line in f:
                if max_examples is not None and len(self.examples) >= max_examples:
                    break
                r = json.loads(line)
                text = messages_to_text(r["messages"], tokenizer)
                ids = tokenizer(
                    text, truncation=True, max_length=max_len,
                    padding="max_length", return_tensors="pt",
                )
                input_ids = ids["input_ids"][0]
                attn = ids["attention_mask"][0]
                # Labels = input_ids with pad tokens set to -100 (ignore)
                labels = input_ids.clone()
                labels[attn == 0] = -100
                self.examples.append({
                    "input_ids": input_ids,
                    "attention_mask": attn,
                    "labels": labels,
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def main():
    device = xm.xla_device()
    world_size = xr.world_size()
    rank = xr.global_ordinal()
    print(f"[rank {rank}/{world_size}] device: {device}")

    # SPMD mesh: single axis across all chips (FSDPv2 style)
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (num_devices, 1)
    device_ids = np.array(range(num_devices))
    mesh = xs.Mesh(device_ids, mesh_shape, ("fsdp", "tensor"))
    print(f"SPMD mesh: {mesh_shape} over {num_devices} devices")

    # ── Tokenizer + model ──────────────────────────────────────
    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model (bf16): {MODEL_ID}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    print(f"  model loaded in {time.time()-t0:.1f}s")

    # ── Move to XLA first (lazy, not yet materialized), then shard ───
    # torch_xla's mark_sharding requires XLA tensors, so order matters:
    # .to(device) makes them lazy XLA tensors; mark_sharding records
    # intent; materialization + shard happens at first op (no OOM).
    model = model.to(device)
    sharded = 0
    for name, param in model.named_parameters():
        if param.ndim == 2 and all(d % num_devices == 0 for d in [param.shape[0]]):
            xs.mark_sharding(param, mesh, ("fsdp", None))
            sharded += 1
    print(f"  sharded {sharded} 2D params across {num_devices} chips (fsdp axis=0)")

    # Gradient checkpointing disabled: at bs=8 seq=512 with params FSDP-sharded,
    # activations per chip are only ~400MB, well under HBM limit. GC doubled
    # forward cost (recompute during backward), dominating step time.
    model.enable_input_require_grads()  # required for PEFT training

    # ── Apply LoRA via PEFT ────────────────────────────────────
    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Data ───────────────────────────────────────────────────
    print(f"Loading training data: {TRAIN_FILE} (cap={MAX_EXAMPLES})")
    ds = JsonlQADataset(TRAIN_FILE, tokenizer, MAX_LEN, max_examples=MAX_EXAMPLES)
    print(f"  {len(ds)} training examples")
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # ── Optimizer ──────────────────────────────────────────────
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01,
    )

    # ── Train ──────────────────────────────────────────────────
    model.train()
    step = 0
    t_train_start = time.time()
    for epoch in range(EPOCHS):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            # bs=8 splits cleanly into 8 FSDP shards, 1 example per chip
            xs.mark_sharding(batch["input_ids"], mesh, ("fsdp", None))
            xs.mark_sharding(batch["attention_mask"], mesh, ("fsdp", None))
            xs.mark_sharding(batch["labels"], mesh, ("fsdp", None))
            out = model(**batch)
            loss = out.loss / GRAD_ACCUM
            loss.backward()
            step += 1
            if step % GRAD_ACCUM == 0:
                optim.step()
                optim.zero_grad()
                # Log every optimizer step — the .item() call forces materialization,
                # so keep it rare. With GRAD_ACCUM=4 this is every 4 micro-steps.
                if (step // GRAD_ACCUM) % LOG_EVERY == 0:
                    xm.master_print(
                        f"  epoch {epoch+1} opt-step {step//GRAD_ACCUM} "
                        f"(micro {step}) loss {(loss.item()*GRAD_ACCUM):.4f} "
                        f"elapsed {time.time()-t_train_start:.1f}s",
                        flush=True,
                    )
                    sys.stdout.flush()
            xm.mark_step()
        xm.master_print(f"✓ epoch {epoch+1}/{EPOCHS} done")

    # ── Save adapter only (small, ~20 MB) ──────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    # Move to CPU before saving to avoid XLA tensor serialization issues
    model = model.to("cpu")
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    xm.master_print(f"✓ adapter saved → {OUT_DIR}")

if __name__ == "__main__":
    main()
