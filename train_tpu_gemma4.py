"""
Gemma 4 31B LoRA fine-tuning on TPU v6e-8 — Phase 2 migration from Gemma 2 27B.

Forked from train_tpu_hf_peft.py. Changes vs Phase 1:
  - MODEL_ID: google/gemma-2-27b-it → google/gemma-4-31B-it (dense 31B, same data).
  - Chat template: Gemma 4 uses `<|turn>…<turn|>` instead of Gemma 2's
    `<start_of_turn>…<end_of_turn>`; we now call tokenizer.apply_chat_template
    so the template stays model-agnostic.
  - Requires transformers >= 5.6.2 (gemma4 was introduced in the 5.x line;
    4.57.6 and earlier raise `KeyError: 'gemma4'` on config load).
  - Layer naming / LoRA target_modules are unchanged (Gemma 4 keeps
    `self_attn.{q,k,v,o}_proj` and `mlp.{gate,up,down}_proj`).
  - Hidden dims are all multiples of 8, so the existing FSDP-axis-0 sharding
    rule applies without modification.

Usage on TPU VM:
    PJRT_DEVICE=TPU python3 train_tpu_gemma4.py
"""
import os
os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("XLA_USE_SPMD", "1")
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

if not hasattr(torch, "xla"):
    torch.xla = torch_xla

import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader

MODEL_ID   = "google/gemma-4-31B-it"
TRAIN_FILE = "/home/zczqrso/train.jsonl"
OUT_DIR    = "/home/zczqrso/gemma4_31b_financial_adapter_hf"
MAX_LEN    = 512
BATCH_SIZE = 4
EPOCHS     = 2
LR         = 1e-4
LORA_RANK  = 8
GRAD_ACCUM = 2
LOG_EVERY  = 1
MAX_EXAMPLES = 160

def messages_to_text(msgs, tokenizer):
    # Gemma (2 and 4) don't support a distinct system role; merge system into user.
    system = next((m["content"] for m in msgs if m["role"] == "system"), "")
    user   = next((m["content"] for m in msgs if m["role"] == "user"), "")
    asst   = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
    merged = [
        {"role": "user", "content": (system + "\n\n" + user) if system else user},
        {"role": "assistant", "content": asst},
    ]
    return tokenizer.apply_chat_template(merged, tokenize=False, add_generation_prompt=False)

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

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (num_devices, 1)
    device_ids = np.array(range(num_devices))
    mesh = xs.Mesh(device_ids, mesh_shape, ("fsdp", "tensor"))
    print(f"SPMD mesh: {mesh_shape} over {num_devices} devices")

    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Use AutoModelForCausalLM -> Gemma4ForConditionalGeneration so the
    # checkpoint keys (prefixed model.language_model.*) match the model tree.
    # Gemma4ForCausalLM alone silently loads zero weights on this repo because
    # its expected prefix is model.* (no language_model). We accept the small
    # HBM cost of carrying the frozen vision + audio towers (~1 GB bf16 total)
    # rather than hand-remap the state dict.
    print(f"Loading model (bf16): {MODEL_ID}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    print(f"  model loaded in {time.time()-t0:.1f}s")

    model = model.to(device)
    sharded = 0
    for name, param in model.named_parameters():
        if param.ndim == 2 and all(d % num_devices == 0 for d in [param.shape[0]]):
            xs.mark_sharding(param, mesh, ("fsdp", None))
            sharded += 1
    print(f"  sharded {sharded} 2D params across {num_devices} chips (fsdp axis=0)")

    model.enable_input_require_grads()

    # Regex-scoped LoRA: only inject into the language_model tower. Plain
    # substring matching on `q_proj` etc. would also hit the vision tower's
    # Gemma4ClippableLinear (1152×1152), which PEFT can't adapt.
    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,
        target_modules=(
            r".*\.language_model\.layers\.\d+\."
            r"(self_attn\.(q|k|v|o)|mlp\.(gate|up|down))_proj$"
        ),
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print(f"Loading training data: {TRAIN_FILE} (cap={MAX_EXAMPLES})")
    ds = JsonlQADataset(TRAIN_FILE, tokenizer, MAX_LEN, max_examples=MAX_EXAMPLES)
    print(f"  {len(ds)} training examples")
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01,
    )

    model.train()
    step = 0
    t_train_start = time.time()
    for epoch in range(EPOCHS):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
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

    os.makedirs(OUT_DIR, exist_ok=True)
    model = model.to("cpu")
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    xm.master_print(f"✓ adapter saved → {OUT_DIR}")

if __name__ == "__main__":
    main()
