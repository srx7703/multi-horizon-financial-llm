# Gemma 4 模型配置指南 — 金融 LLM Agent 项目

> 针对 SEC S-1 招股书知识蒸馏 + **Gemma 4 31B Dense** 微调场景  
> 基于 **Google Cloud TPU v5/v6** 环境（2026 年最优配置）

---

## 1. 模型选择建议

| 模型 | 参数量 | 类型 | Arena 排名 | 推荐度 | 说明 |
|------|--------|------|-----------|--------|------|
| Gemma 2 9B | 9B | Dense | - | ⭐⭐⭐ | 成本低，但性能已过时 |
| Gemma 4 E4B | 4B | Effective | - | ⭐ | 太小，金融领域不足 |
| Gemma 4 26B | 26B | MoE | #6 | ⭐⭐⭐⭐ | MoE 架构，LoRA 配置复杂 |
| **Gemma 4 31B** | **31B** | **Dense** | **#3（开源最强）** | **⭐⭐⭐⭐⭐** | **推荐。** 性能最强，LoRA 配置简单，TPU v5/v6 充分利用 |

### ✅ 推荐：`google/gemma-4-31b-it`（指令微调版本）

- **为什么选 31B**：Arena 排名 #3（仅次于闭源模型），金融风险分析能力远超 Gemma 2 9B
- **为什么选 -it 版本**：Gemma 4 31B-it 已基础指令微调，与我们的蒸馏数据方向一致，降低微调冲突风险
- **密集架构优势**：所有参数都激活，LoRA 微调配置标准化，无需处理 MoE expert 路由复杂性
- **TPU v5/v6 匹配**：31B 是为高端 TPU 设计的，你的硬件配置完全足够（v5 单 pod 32GB HBM，v6 更高）

---

## 2. 下载方式

### 方式 A：Hugging Face（推荐）

```bash
# 1. 安装依赖
pip install transformers accelerate huggingface_hub torch

# 2. 登录 Hugging Face（需先在网页端接受 Google 使用协议）
#    前往 https://huggingface.co/google/gemma-4-31b-it 点击 "Agree and access repository"
huggingface-cli login
# 输入你的 HF Token（在 https://huggingface.co/settings/tokens 生成）

# 3. 下载模型（Python）
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "google/gemma-4-31b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
# 注意：31B 模型很大（~62GB），需要足够的磁盘空间
# 如果在本地开发环境，可用 load_in_4bit 或在 TPU 上直接加载
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="bfloat16",
    device_map="auto"  # 自动分配设备（GPU/CPU/多GPU）
)
```

### 方式 B：Google Vertex AI（TPU 官方路径）

```bash
# Vertex AI 集成了 Gemma 4，可直接在 TPU 上加载，无需下载
# 如果使用 GCP 上的 TPU VM，推荐此方式
pip install google-cloud-aiplatform
```

```python
from google.cloud import aiplatform

# 在 Vertex AI 上加载模型（自动 TPU 优化）
aiplatform.init(project="your-project", location="us-central1")

# 详见第 3 节 Keras + JAX 方式
```

### 方式 C：Keras Hub（TPU 推荐路径）⭐

```bash
pip install keras keras-hub tensorflow-hub
```

```python
import os
os.environ["KERAS_BACKEND"] = "jax"  # TPU 必须用 JAX 后端

import keras_hub

# 加载 Gemma 4 31B（Keras Hub 自动优化 TPU 内存分配）
model = keras_hub.models.GemmaCausalLM.from_preset("gemma4_31b_it_en")
```

---

## 3. 训练框架选择（TPU v5/v6 环境）

### ⚠️ 关键限制：QLoRA 不支持 TPU

**bitsandbytes 库（QLoRA 依赖）仅支持 NVIDIA GPU**，无法在 TPU 上运行。TPU 上必须使用标准 LoRA。

> **好消息**：TPU v5/v6 每个 pod 有 32GB+ HBM 显存，LoRA 微调 Gemma 4 31B 完全充足，无需量化。

### 推荐路径对比

| 方案 | 框架 | TPU 支持 | 分布式 | 推荐度 |
|------|------|---------|--------|--------|
| **Keras + JAX** | Keras 3 + JAX | ✅ 原生 | ✅ pjit/shard | **⭐⭐⭐⭐⭐** |
| MaxText | 纯 JAX | ✅ 原生 | ✅ SPMD | ⭐⭐⭐⭐ |
| HF + optimum-tpu | Transformers | ✅ 可用 | ⚠️ 有限 | ⭐⭐⭐ |

### 推荐：Keras + JAX（Google 官方 + 最稳定）

Gemma 4 31B 针对 Keras 做了优化。官方教程：
https://ai.google.dev/gemma/docs/core/lora_tuning

**TPU v5/v6 特定优化**：
- 原生 bf16（v5/v6 最优精度）
- pjit 自动分布式（多 pod 自动并行）
- 内存优化（Gemma 4 31B 适配了 HBM 分配）

---

## 4. LoRA 参数配置（Gemma 4 31B）

### 4.1 Keras + JAX 版本（TPU v5/v6 推荐）⭐

```python
import os
os.environ["KERAS_BACKEND"] = "jax"

import jax
import keras
import keras_hub

# ============ 1. 加载模型 ============
model = keras_hub.models.GemmaCausalLM.from_preset("gemma4_31b_it_en")

# ============ 2. 启用 LoRA ============
model.backbone.enable_lora(
    rank=16,           # 金融领域知识注入需要足够容量
    lora_alpha=32,     # = 2 * rank
)

# ============ 3. 设置混合精度（TPU 原生 bf16） ============
keras.mixed_precision.set_global_policy('mixed_bfloat16')

# ============ 4. 编译模型 ============
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.AdamW(
        learning_rate=5e-5,    # LoRA 推荐范围
        weight_decay=0.01,     # 正则化，防止过拟合
        clipnorm=1.0,          # 梯度裁剪
    ),
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
    jit_compile=True,  # JAX JIT 加速（TPU 必须）
)

# ============ 5. 训练 ============
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=3,
    batch_size=2,  # 根据 TPU pod 内存调整（v5 一般为 2-4）
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=1,  # 验证集不改进则停止
            restore_best_weights=True,
        ),
    ],
)
```

### 4.2 Hugging Face PEFT 版本（GPU 备选方案）

```python
from peft import LoraConfig, TaskType, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(base_model, lora_config)
print(model.print_trainable_parameters())
# 预期：trainable params ~70M (Gemma 4 31B) || trainable% ~0.22%
```

### 4.3 参数选择理由（针对 2000 条小数据集）

| 参数 | 值 | 理由 |
|------|-----|------|
| `r=16` | 16 | Gemma 4 31B 参数多，r=16 是金融知识注入的最小值。r=8 太小会丢失蒸馏知识 |
| `lora_alpha=32` | 2×r | 标准 heuristic，学习率缩放稳定 |
| `target_modules` | 全部 7 个 | Dense 模型，全模块微调适配效果最好 |
| `lora_dropout=0.05` | 0.05 | **关键**：2000 条小数据，0.05 防过拟合。不要用 0 |
| `clipnorm=1.0` | 1.0 | 梯度裁剪，大模型训练稳定性必须项 |
| `early_stopping` | patience=1 | 验证集 loss 不改进立即停止，防过拟合 |

---

## 5. 训练超参数（Gemma 4 31B + TPU v5/v6）

### Keras 版本（推荐）

Keras 训练配置已集成在上面 4.1 节，这里补充分布式训练配置：

```python
import os
os.environ["KERAS_BACKEND"] = "jax"

import jax
import keras
import keras_hub

# ============ 分布式训练配置（多 pod） ============
# 如果使用 TPU pod slice（e.g., v5-32），启用自动分布式
if jax.device_count() > 8:
    # 自动使用所有可用 TPU 核心
    strategy = keras.distribution.DataParallelism()
    with strategy.scope():
        model = keras_hub.models.GemmaCausalLM.from_preset("gemma4_31b_it_en")
        model.backbone.enable_lora(rank=16)
        model.compile(...)
else:
    # 单 pod 训练（v5-8）
    model = keras_hub.models.GemmaCausalLM.from_preset("gemma4_31b_it_en")
    model.backbone.enable_lora(rank=16)
    model.compile(...)

# ============ 训练参数 ============
training_config = {
    "epochs": 3,                           # 2000 条数据，3 轮足够
    "batch_size": 2,                       # v5-8: 2, v5-16+: 可增至 4
    "learning_rate": 5e-5,                 # LoRA 推荐值
    "weight_decay": 0.01,                  # L2 正则化
    "warmup_steps": int(2000 / 2 * 0.1),  # 10% warmup
    "max_grad_norm": 1.0,                  # 梯度裁剪
    "logging_steps": 10,
    "eval_steps": 50,                      # 每 50 step 验证一次
}
```

### HF Transformers 版本（GPU 备选）

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./gemma4-finance-lora",
    
    # === 核心超参数 ===
    num_train_epochs=3,
    per_device_train_batch_size=1,         # 31B 模型，GPU 需要很小的 batch
    gradient_accumulation_steps=16,        # 有效 batch_size = 1 * 16 = 16
    learning_rate=5e-5,
    
    # === 学习率调度 ===
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    
    # === 精度与优化 ===
    bf16=True,                             # A100/H100 支持 bf16
    fp16=False,
    
    # === 正则化 ===
    weight_decay=0.01,
    max_grad_norm=1.0,
    
    # === 序列长度 ===
    # max_seq_length=2048                  # 在 SFTTrainer 中设置
    
    # === 日志与保存 ===
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    
    # === 其他 ===
    remove_unused_columns=False,
    report_to="wandb",                     # 推荐用 W&B 可视化 31B 训练
)
```

### 序列长度建议

| 数据类型 | 建议 max_seq_length | 说明 |
|---------|---------------------|------|
| 简短 QA 对（< 500 tokens） | 1024 | 节省内存 |
| **包含 S-1 摘录的 QA 对（推荐）** | **2048** | 标准金融问答长度 |
| 长文档分析 | 4096 | 31B 有足够内存，可用 4K |

### 防过拟合策略（2000 条小数据集）

| 策略 | 配置 | 理由 |
|------|------|------|
| **Early Stopping** | patience=1 | 验证集 loss 不改进立即停止 |
| **Dropout** | lora_dropout=0.05 | 5% 概率丢弃激活 |
| **Weight Decay** | 0.01 | L2 正则化，防止权重过大 |
| **Learning Rate** | 5e-5 | 保守学习率，避免快速过拟合 |
| **Validation Split** | 15% (300 条) | 每个 epoch 验证一次 |
| **Gradient Accumulation** | 根据 batch size 调整 | 保持有效 batch >= 16 |

---

## 6. 数据格式化（Gemma 4 对话模板）

### Gemma 4 原生对话模板

```
<bos><start_of_turn>user
{问题内容}<end_of_turn>
<start_of_turn>model
{回答内容}<end_of_turn>
```

**关键点**：
- 使用 `<start_of_turn>user` / `<start_of_turn>model`（不支持 system role）
- 系统指令需拼接到第一个 user turn
- 每个 turn 以 `<end_of_turn>` 结尾
- 模型回答前缀为 `<start_of_turn>model`

### JSONL 数据格式（推荐）

```json
{"messages": [
    {"role": "user", "content": "You are a senior financial analyst specializing in SEC S-1 IPO filings and Wall Street risk assessment.\n\nBased on the SEC S-1 filing excerpt below, provide a concise financial risk analysis in professional research report style.\n\n[S-1 FILING EXCERPT]\n\nQuestion: What are the material financial and operational risks disclosed in this filing?"}, 
    {"role": "assistant", "content": "[华尔街研报风格的分析，包括风险等级、量化指标、缓解措施等]"}
]}
```

### Keras 训练数据加载

```python
import json

def load_messages_from_jsonl(jsonl_path, max_seq_len=2048):
    """加载 JSONL 并构建 Gemma 4 格式文本"""
    texts = []
    with open(jsonl_path, 'r') as f:
        for idx, line in enumerate(f):
            item = json.loads(line)
            msgs = item["messages"]
            
            # 构建 Gemma 4 对话格式
            text = "<bos>"
            for msg in msgs:
                role = "user" if msg["role"] == "user" else "model"
                text += f"<start_of_turn>{role}\n{msg['content']}<end_of_turn>\n"
            
            # 截断到 max_seq_len（防止内存溢出）
            if len(text) > max_seq_len * 4:  # 粗估算：1 token ≈ 4 字符
                text = text[:max_seq_len * 4]
            
            texts.append(text)
    
    return texts

# 加载数据
train_texts = load_messages_from_jsonl("gemma4_train.jsonl")
val_texts = load_messages_from_jsonl("gemma4_val.jsonl")

print(f"✅ 训练集: {len(train_texts)} 条")
print(f"✅ 验证集: {len(val_texts)} 条")
```

### 将蒸馏产出转换为 messages 格式

```python
import json

def convert_distilled_qa_to_gemma4(input_file, output_file):
    """将 Gemini 蒸馏产出的 QA 对转换为 Gemma 4 训练格式"""
    
    system_instruction = (
        "You are an expert financial analyst specializing in SEC S-1 filings and IPO risk assessment. "
        "Provide concise, quantitative analysis in professional Wall Street research report style. "
        "Focus on material risks, financial metrics, and mitigation strategies."
    )

    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for idx, line in enumerate(f_in):
            try:
                item = json.loads(line)
                
                # 适配不同字段名
                question = item.get("question") or item.get("input") or item.get("prompt", "")
                answer = item.get("answer") or item.get("output") or item.get("response", "")
                context = item.get("context") or ""
                
                if not question or not answer:
                    print(f"⚠️  Skip line {idx}: missing question or answer")
                    continue
                
                # 构建完整问题（系统指令 + 上下文 + 问题）
                full_question = f"{system_instruction}\n\n"
                if context:
                    full_question += f"[SEC S-1 FILING EXCERPT]\n{context}\n\n"
                full_question += f"Question: {question}"
                
                formatted = {
                    "messages": [
                        {"role": "user", "content": full_question},
                        {"role": "assistant", "content": answer}
                    ]
                }
                
                f_out.write(json.dumps(formatted, ensure_ascii=False) + "\n")
                
            except Exception as e:
                print(f"⚠️  Error processing line {idx}: {e}")
                continue

# 执行转换
convert_distilled_qa_to_gemma4(
    "training_dataset.jsonl",      # Gemini 蒸馏产出
    "gemma4_train.jsonl"            # Gemma 4 训练格式
)
print("✅ 数据转换完成！")
```

---

## 7. 完整训练脚本（TPU v5/v6 — Keras + JAX）⭐

```python
"""
Gemma 4 31B LoRA Fine-Tuning on Google Cloud TPU v5/v6
金融 LLM Agent — SEC S-1 知识蒸馏数据微调
作者: Financial LLM Team
日期: 2026-04
"""
import os
os.environ["KERAS_BACKEND"] = "jax"

import json
import sys
import keras
import keras_hub
import jax
from datetime import datetime

print(f"🔧 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📊 TPU 核心数: {jax.device_count()}")
print(f"💾 TPU 类型: {jax.devices()[0].device_kind}")

# =====================
# 1. 加载模型
# =====================
print("\n⏳ 加载 Gemma 4 31B 模型...")
model = keras_hub.models.GemmaCausalLM.from_preset("gemma4_31b_it_en")
tokenizer = model.preprocessor.tokenizer
print(f"✅ 模型加载完成，参数量: ~31B")

# =====================
# 2. 启用 LoRA
# =====================
print("\n⏳ 启用 LoRA (rank=16)...")
model.backbone.enable_lora(rank=16)
print("✅ LoRA 启用完成")

# 获取可训练参数数量
total_params = model.count_params()
print(f"   总参数: {total_params:,}")

# =====================
# 3. 设置混合精度（TPU 原生 bf16）
# =====================
keras.mixed_precision.set_global_policy('mixed_bfloat16')
print("✅ 混合精度设置: bfloat16")

# =====================
# 4. 加载并格式化数据
# =====================
print("\n⏳ 加载训练数据...")

def load_messages_from_jsonl(jsonl_path, max_size=None):
    """从 JSONL 加载消息并转换为 Gemma 4 格式"""
    texts = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if max_size and idx >= max_size:
                break
            try:
                item = json.loads(line)
                msgs = item.get("messages", [])
                if not msgs:
                    continue
                
                # 构建 Gemma 4 对话格式
                text = "<bos>"
                for msg in msgs:
                    role = "user" if msg["role"] == "user" else "model"
                    text += f"<start_of_turn>{role}\n{msg['content']}<end_of_turn>\n"
                
                texts.append(text)
            except Exception as e:
                print(f"   ⚠️  跳过第 {idx} 行: {e}")
                continue
    
    return texts

# 加载数据
try:
    train_texts = load_messages_from_jsonl("gemma4_train.jsonl")
except FileNotFoundError:
    print("❌ 错误: 找不到 gemma4_train.jsonl")
    print("   请确保文件路径正确，或运行数据转换脚本")
    sys.exit(1)

# 划分训练/验证集（85/15）
split_idx = int(len(train_texts) * 0.85)
train_data = train_texts[:split_idx]
val_data = train_texts[split_idx:]

print(f"✅ 训练集: {len(train_data)} 条样本")
print(f"✅ 验证集: {len(val_data)} 条样本")

# 显示样本长度统计
avg_len = sum(len(t) for t in train_texts) / len(train_texts)
print(f"   平均长度: {avg_len:.0f} 字符 (~{avg_len/4:.0f} tokens)")

# =====================
# 5. 编译模型
# =====================
print("\n⏳ 编译模型...")
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.AdamW(
        learning_rate=5e-5,
        weight_decay=0.01,
        clipnorm=1.0,
    ),
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
    jit_compile=True,  # JAX JIT 加速
)
print("✅ 编译完成")

# =====================
# 6. 定义回调函数
# =====================
class LogCallback(keras.callbacks.Callback):
    """自定义日志回调"""
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"\n📊 Epoch {epoch+1} 结果:")
        print(f"   Train Loss: {logs.get('loss', 0):.4f}")
        print(f"   Val Loss:   {logs.get('val_loss', 0):.4f}")

callbacks = [
    LogCallback(),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=1,
        restore_best_weights=True,
        verbose=1,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath="./checkpoints/gemma4-finance-epoch-{epoch:02d}.weights.h5",
        monitor='val_loss',
        save_best_only=True,
        verbose=1,
    ),
]

# =====================
# 7. 训练
# =====================
print("\n🚀 开始训练...")
print(f"   Batch Size: 2")
print(f"   Epochs: 3")
print(f"   学习率: 5e-5")
print(f"   LoRA Rank: 16\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=3,
    batch_size=2,
    callbacks=callbacks,
    verbose=1,
)

# =====================
# 8. 保存模型和 LoRA 权重
# =====================
print("\n💾 保存模型...")

# 保存完整模型
model.save("./gemma4-finance-agent-final")
print("✅ 完整模型已保存到 ./gemma4-finance-agent-final")

# 分别保存 LoRA 权重（用于后续推理）
model.backbone.save_lora_weights("./gemma4-finance-lora-weights")
print("✅ LoRA 权重已保存到 ./gemma4-finance-lora-weights")

# 保存 tokenizer
tokenizer.save_preset("./gemma4-finance-tokenizer")
print("✅ Tokenizer 已保存到 ./gemma4-finance-tokenizer")

# =====================
# 9. 训练统计
# =====================
print(f"\n✨ 训练完成！")
print(f"   结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"   最终 Validation Loss: {history.history['val_loss'][-1]:.4f}")
print(f"\n📂 模型位置: ./gemma4-finance-agent-final")
print(f"📂 LoRA 权重: ./gemma4-finance-lora-weights")
```

### 运行脚本

```bash
# GCP TPU VM 上运行
cd /path/to/project

# 激活虚拟环境
source sec_env/bin/activate

# 确保依赖安装
pip install keras keras-hub tensorflow-hub jax jaxlib -U

# 运行训练脚本
python train_gemma4_tpu.py
```

---

## 8. 完整训练脚本（GPU — HF Transformers + PEFT 备选方案）

```python
"""
Gemma 2 9B QLoRA Fine-Tuning on GPU
仅当无法使用 TPU 时的备选方案
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset

# =====================
# 1. 量化配置（仅 GPU）
# =====================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# =====================
# 2. 加载模型
# =====================
model_id = "google/gemma-2-9b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

# =====================
# 3. LoRA 配置
# =====================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# =====================
# 4. 加载数据
# =====================
dataset = load_dataset("json", data_files="gemma_train.jsonl", split="train")
dataset = dataset.train_test_split(test_size=0.15, seed=42)

# =====================
# 5. 训练配置
# =====================
training_args = TrainingArguments(
    output_dir="./gemma-finance-lora",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    weight_decay=0.01,
    max_grad_norm=1.0,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
)

# =====================
# 6. 开始训练
# =====================
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=training_args,
    peft_config=lora_config,
    max_seq_length=2048,
    tokenizer=tokenizer,
)

trainer.train()

# =====================
# 7. 保存
# =====================
trainer.save_model("./gemma-finance-lora-final")
tokenizer.save_pretrained("./gemma-finance-lora-final")
print("训练完成，模型已保存！")
```

---

## 9. 参数速查表（Gemma 4 31B + TPU v5/v6）

| 类别 | 参数 | 推荐值 | 备注 |
|------|------|--------|------|
| **模型** | 模型选择 | `google/gemma-4-31b-it` | 指令微调版本 |
| | Preset (Keras) | `gemma4_31b_it_en` | Keras Hub 预设 |
| **LoRA** | rank (r) | 16 | 31B 大模型，16 是最小值 |
| | alpha | 32 | 2 × r（标准 heuristic） |
| | dropout | 0.05 | 关键：防止 2000 条小数据过拟合 |
| | target_modules | 全部 7 个 | Dense 模型全模块微调 |
| **训练** | learning_rate | 5e-5 | LoRA 保守范围 |
| | epochs | 3 | 小数据集防过拟合 |
| | batch_size | 2 (v5-8) / 4 (v5-16+) | 根据 TPU 内存调整 |
| | max_seq_length | 2048 | 金融 QA 标准长度 |
| | warmup | 10% | 前 10% 步数线性 warmup |
| | scheduler | cosine | Cosine 退火 |
| | weight_decay | 0.01 | L2 正则化 |
| | max_grad_norm | 1.0 | 梯度裁剪 |
| **防过拟合** | early_stopping | patience=1 | 验证集不改进立即停止 |
| | val_split | 15% (300 条) | 每 epoch 验证 |
| **精度** | TPU 精度 | bfloat16 | v5/v6 原生精度 |
| | 框架 | Keras + JAX | Google 官方推荐 |
| **TPU** | 推荐配置 | v5-8 | 8 核约 32GB HBM，31B LoRA 足够 |
| | 大规模 | v5-32 或 v6 | 自动分布式数据并行 |
| | 环境变量 | `KERAS_BACKEND=jax` | 必须设置 |

---

## 10. 常见问题（Gemma 4 版本）

**Q: 为什么选 Gemma 4 31B 而不是 26B MoE？**
A: 
- 31B 性能排名开源 #3（vs 26B 的 #6）
- Dense 架构对 LoRA 配置更简单（MoE 需要处理 expert 路由）
- 你有 TPU v5/v6，参数容量不是瓶颈
- 从 Gemini 蒸馏的知识注入到最强模型效果最优

**Q: 为什么用 -it 版本而非 base？**
A: 
- Gemma 4 31B-it 已有基础指令微调，与金融蒸馏数据方向一致
- 避免与完全 base 模型冲突（base 需要完全重新训练指令能力）
- -it 版本微调更稳定，防止灾难遗忘（catastrophic forgetting）

**Q: 2000 条数据真的够吗？**
A: 
- **足够**，但需要严格防过拟合
- 使用 LoRA rank=16（不要更高）+ dropout=0.05 + early stopping
- 监控验证集 loss，patience=1 （不改进立即停止）
- 预期能吸收 80-90% 的蒸馏知识

**Q: 能否增加数据量到 5000+ 条后再训练？**
A: 
- **建议先用 2000 条跑通流程**，验证效果
- 之后可逐步增加数据，微调会更稳定（rank 可增至 32）
- Gemma 4 31B 有足够容量吸收 5000+ 条金融 QA

**Q: TPU v5 vs v6，选哪个？**
A: 
- **v5** 足够用（成本较低），v5-8 就可以训练 31B LoRA
- **v6** 更快但成本高，如果公司有预算优先用 v6
- 两者都支持本配置，改变主要是训练速度

**Q: 怎么从微调后的模型做推理？**
A: 
```python
import keras_hub
model = keras_hub.models.GemmaCausalLM.from_preset("gemma4_31b_it_en")
model.backbone.load_lora_weights("./gemma4-finance-lora-weights")

# 推理
prompt = "<bos><start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n"
output = model.generate(prompt, max_length=512)
```

**Q: 怎么合并 LoRA 权重到基础模型？**
A: 
- 标准 LoRA 权重可通过 `model.backbone.load_lora_weights()` 加载后直接推理
- 如需完全合并（用于部署），Keras 暂未提供官方接口
- 推荐保存 LoRA 权重和基础模型分开，推理时动态加载

---

## 检查清单

**训练前**：
- [ ] TPU 环境已准备（v5-8 或更大）
- [ ] Keras + JAX + keras-hub 已安装最新版本
- [ ] 数据已转换为 `messages` 格式 JSONL
- [ ] 验证数据集大小（训练 ~1700 条，验证 ~300 条）

**训练中**：
- [ ] 监控训练日志，验证 loss 下降
- [ ] 检查内存使用（不应超过 TPU HBM 容量）
- [ ] 验证 LoRA 参数被正确激活

**训练后**：
- [ ] 检查模型已保存到 `./gemma4-finance-agent-final`
- [ ] LoRA 权重已保存到 `./gemma4-finance-lora-weights`
- [ ] 在验证集上评估最终性能

---

## 参考资料

- [Gemma 4 Official Release (Google DeepMind)](https://deepmind.google/models/gemma/gemma-4/)
- [Gemma 4 Fine-Tuning Guide (Google AI)](https://ai.google.dev/gemma/docs/core/lora_tuning)
- [Keras LoRA Documentation](https://keras.io/examples/keras_recipes/parameter_efficient_finetuning_of_gemma_with_lora_and_qlora/)
- [Gemma 4 on Hugging Face](https://huggingface.co/google/gemma-4-31b-it)
- [TPU Performance Tuning (Google Cloud)](https://cloud.google.com/tpu/docs/performance-guide)
- [JAX Documentation](https://jax.readthedocs.io/)
