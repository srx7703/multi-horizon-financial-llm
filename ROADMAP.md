# Project Roadmap — Multi-Horizon Financial Research Agent

> 用途：GitHub portfolio（不参加 hackathon，不做 demo video）
> 最后更新：2026-04-23
> 当前版本：已完成数据扩展 + Gemma 2 27B v1/v2 adapter 训练，正准备评估 + 打磨

---

## 当前状态（已完成）

- ✅ 23 份 10-K summary（5 家公司 × 5 年）
- ✅ 136 份 10-Q summary（69 家 × 近 2 季度，通过 `sec_expand.py`）
- ✅ 222 份 8-K summary（69 家 × 近 90 天）
- ✅ 1325 条 QA 训练数据（`finetune_data_v2/`）
- ✅ Streamlit app + Vertex AI Vector Search RAG pipeline
- ✅ Gemma 4 E4B 本地 LoRA adapter（MLX，旧版）
- ✅ **Gemma 2 27B + v1 adapter**（HF 通用金融 QA 数据训练，TPU v6e-8）
- ✅ **Gemma 2 27B + v2 adapter**（SEC 自有数据 1060 QA × 2 epochs 训练）
- ✅ TPU v6e-8 训练 pipeline（model-parallel sharding，bfloat16，rank=8 LoRA）

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

## Phase 2 — Gemma 4 27B 升级（目标：3-5 天，~6-8 小时）

### 目标
展示 pipeline 的可迁移性，扩大 BERTScore 对比到四方。打 git tag `v2.0-gemma4`。

### 任务清单

| # | 任务 | 预估 | 风险 | 备注 |
|---|---|---|---|---|
| 1 | 启动 TPU，验证 Gemma 4 27B keras_hub preset | 30min | 中 | preset 不存在则切换 HF transformers 路径 |
| 2 | 适配 sharding layout regex（Gemma 4 层命名可能变） | 1-2h | 中 | 看 `model.summary()` 调 regex |
| 3 | 同一份 v2 数据训练 Gemma 4 27B | 1h TPU | 低 | 复用 Phase 1 的 train_gemma.py |
| 4 | 四方 BERTScore（base2 / 2+v2 / base4 / 4+v2） | 1h | 低 | 扩展 Phase 1 eval 脚本 |
| 5 | README 加 "Gemma 2 → Gemma 4 Migration" 章节 | 1h | — | 讲迁移决策 |
| 6 | ARCHITECTURE.md 补 "Why Gemma 4" 章节 | 30min | — | trade-off 叙事 |

**Phase 2 交付物**：
- `git tag v2.0-gemma4`
- README 变成"双模型对比"叙事
- ARCHITECTURE.md 加 migration 章节

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

## 下一步操作

当前就绪可做的任务按优先级：
1. **[P0] Phase 1 任务 ①**: BERTScore 三方对比评估
2. [P1] Phase 1 任务 ②: README 大改
3. [P1] Phase 1 任务 ③: ARCHITECTURE.md

---

## 修订历史

- 2026-04-23: 初版，定义 Phase 1 + Phase 2，明确 GitHub portfolio 定位
