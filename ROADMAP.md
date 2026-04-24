# Project Roadmap — Multi-Horizon Financial Research Agent

> 用途：GitHub portfolio（不参加 hackathon，不做 demo video）
> 最后更新：2026-04-24
> 当前版本：Phase 1 (`v1.0-gemma2`) + Phase 2 (`v2.0-gemma4`) 均已完成，四方 BERTScore 对比已出

---

## 当前状态（已完成）

- ✅ 23 份 10-K summary（5 家公司 × 5 年）
- ✅ 136 份 10-Q summary（69 家 × 近 2 季度，通过 `sec_expand.py`）
- ✅ 222 份 8-K summary（69 家 × 近 90 天）
- ✅ 1325 条 QA 训练数据（`finetune_data_v2/`）
- ✅ Streamlit app + Vertex AI Vector Search RAG pipeline
- ✅ Gemma 4 E4B 本地 LoRA adapter（MLX，旧版）
- ✅ **Phase 1 — Gemma 2 27B + v2 adapter**（SEC 数据 1060 QA × 2 epochs，TPU v6e-8 FSDPv2，BERTScore F1 +3.50%，tag `v1.0-gemma2`）
- ✅ **Phase 2 — Gemma 4 31B + v2 adapter**（同一份数据，同一套 LoRA 配置，BERTScore F1 +5.76%，wins 20/20，tag `v2.0-gemma4`）
- ✅ TPU v6e-8 训练 pipeline（model-parallel sharding，bfloat16，rank=8 LoRA）
- ✅ 四方 BERTScore + 两组 paired t-test（`evaluation_results_phase2.json`）
- ✅ Streamlit RAG 演示 GIF（`docs/demo.gif`）

---

## 策略：两阶段推进

**核心原则**：先把 Gemma 2 27B 打磨成独立可交付作品，再做 Gemma 4 27B 升级作为第二章叙事。任何一个阶段没做完不开下一阶段。

---

## Phase 1 — Gemma 2 27B 打磨（目标：3-5 天，~8-9 小时）

### 目标
一个"单独拿出来就能当 portfolio"的完整版本，打 git tag `v1.0-gemma2`。

### 任务清单（按执行顺序）

| # | 任务 | 预估 | 为什么优先 | 产出 |
|---|---|---|---|---|
| 1 | **BERTScore 三方对比** | 2h | 没数字的 ML 项目 = 没说服力 | `evaluation_v2.json` + 表格 |
| 2 | **README 大改** | 2h | HR 第一眼看的就是 README | 新 README.md |
| 3 | **ARCHITECTURE.md** | 2h | 面试官必点进去看的 trade-off 叙事文档 | 独立文档 |
| 4 | **Repo 清理 + 规范 commits** | 1.5h | `git log` 反映工程素养 | 干净目录结构 |
| 5 | **pytest + CI badge** | 1.5h | 绿色 ✓ 提升可信度 | `.github/workflows/ci.yml` |
| 6 | **Streamlit 10s GIF** | 30min | 让 README 有"生命感"（不是 Loom 视频） | `docs/demo.gif` |

**Phase 1 交付物**：
- `git tag v1.0-gemma2`
- GitHub release notes
- README 可以独立讲完整故事

### 关键技术决策点

**BERTScore 评估在哪里跑？**（27B 本地 Mac 跑不动）

选项 A（推荐）：临时起 TPU 跑评估
- 时间：~1 小时（起 TPU + 跑 10 题 × 3 adapter + 停机）
- 成本：~$7 Spot TPU
- 优点：和训练同环境

选项 B：部署到 Vertex AI endpoint
- 时间：~3 小时
- 成本：按调用计费
- 优点：Streamlit 也能用
- 缺点：hack-only 项目不值得

→ **Phase 1 选 A，Phase 2 评估时再考虑 B**

---

## Phase 2 — Gemma 4 31B 升级（已完成 ✅）

### 目标
展示 pipeline 的可迁移性，扩大 BERTScore 对比到四方。打 git tag `v2.0-gemma4`。

### 实际做法 vs 原计划

- 原计划 Gemma 4 **27B**；实际没有 4.x 的 27B dense 权重，选了 **31B dense**（4.x 线上最接近的 apples-to-apples 变种）。
- 原计划 keras_hub preset 起步；实际走 HF `transformers` + PEFT 路径（transformers ≥ 5.6.2 支持 `gemma4`，和 Phase 1 同一个训练 loop）。
- Sharding / LoRA rank / batch / optimizer 全部沿用 Phase 1 —— 改动只有 MODEL_ID、chat template、LoRA target_modules 的 regex scoping 到 `.language_model.` tower。

### 实际结果

| # | 任务 | 结果 |
|---|---|---|
| 1 | Gemma 4 31B 训练（同数据同配置） | ✅ adapter 228 MB |
| 2 | Gemma 4 base n=20 preds | ✅ `preds/preds_gemma4_base.json` |
| 3 | Gemma 4 + v2 adapter n=20 preds | ✅ `preds/preds_gemma4_v2g4.json` |
| 4 | 四方 BERTScore + paired t-test | ✅ `evaluation_results_phase2.json` |
| 5 | README "Gemma 2 → Gemma 4 Migration" 章节 | ✅ |
| 6 | ARCHITECTURE.md Decision 7 "Why Gemma 4" | ✅ |

### 关键数字

| Model | F1 | Δ vs base |
|---|---:|---:|
| Base Gemma 2 27B | 0.8078 | — |
| Gemma 2 27B + v2 | 0.8361 | +3.50% |
| Base Gemma 4 31B | 0.8283 | — |
| **Gemma 4 31B + v2** | **0.8760** | **+5.76%** (t=10.42, wins 20/20) |

### Phase 2 踩坑（写在 ARCHITECTURE §7）

- transformers 5.6.2 `StaticCache.__init__` bug：`num_kv_shared_layers=0` 触发 `layer_types[:-0]` → `[]`，绕过做法在 `generate_tpu_gemma4.py:40-61`
- DynamicCache 会让 XLA 每步 decode 都重编译 → 31B 下直接挂死；改成 StaticCache + prompt left-pad 到 MAX_PROMPT=256

**Phase 2 交付物（已完成）**：
- ✅ `git tag v2.0-gemma4`
- ✅ README 双模型对比叙事（4-way 表 + Phase 2 章节）
- ✅ ARCHITECTURE.md Decision 7 migration 章节

---

## 不做的事（明确 scope out）

- ❌ Loom/YouTube narrated demo video — GitHub portfolio 不需要
- ❌ 部署到 Vertex AI endpoint（除非 Phase 2 评估需要）
- ❌ Hackathon pitch slides / 3 分钟 pitch
- ❌ 写个人博客（可选，不在主线）
- ❌ 扩展到更多公司 / 更多年份（69 家 × 三种 filing 已够用）

---

## 预算估算

| 阶段 | TPU 时间 | 成本 | 备注 |
|---|---|---|---|
| Phase 1 eval | ~1h | ~$7 | Spot v6e-8 |
| Phase 2 训练 + eval | ~2h | ~$14 | Spot v6e-8 |
| Gemini API（QA 生成已完成） | 0 | 0 | — |
| **合计** | **~3h** | **~$21** | 都在 GCP 试用额度里 |

---

## Git 分支 / Tag 策略

- `main` 保持可用状态
- Phase 1 完成 → `git tag v1.0-gemma2` + GitHub release
- Phase 2 完成 → `git tag v2.0-gemma4` + GitHub release
- 失败的实验分支不 push 到 main

---

## 简历 / Portfolio 描述模板（Phase 1 完成后可用）

> **Multi-Horizon Financial Research Agent** — Hierarchical RAG system over 69 S&P 500 companies' SEC filings (10-K + 10-Q + 8-K), combining Vertex AI Vector Search, Gemini 3.1 Pro synthesis, and LoRA fine-tuned Gemma 2 27B trained on 1,060 knowledge-distilled QA pairs on TPU v6e-8. Demonstrated +X% BERTScore F1 improvement via domain-specific knowledge distillation.

（X% 数字等 Phase 1 ① 完成后填入）

---

## 下一步操作（Phase 1 + 2 均已交付）

可选后续（均非必需，portfolio 已完整）：
1. 把 Phase 2 的代码改动 commit 到 main（目前只在工作区），然后 `git push origin v2.0-gemma4`
2. 在 GitHub Releases 页面为两个 tag 写 release note
3. （可选）扩大 BERTScore 测试集到 n=100 或使用 human-labeled QA —— 当前 n=20 的 CI 已不跨零，扩大主要是提高样本可信度而非结论方向

---

## 修订历史

- 2026-04-23: 初版，定义 Phase 1 + Phase 2，明确 GitHub portfolio 定位
