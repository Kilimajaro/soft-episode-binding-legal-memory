# BIMS 法律咨询领域记忆系统测试报告

> **投稿主结果请以 [`REPORT_SCALED.md`](REPORT_SCALED.md) 与 `results/legal_scaled/` 为准**
> （$M{=}400$ 会话、$S{=}300$ 查询；双协议 QA 论文口径 **P1=120 + P2=150 → N=270/库**）。
> 本文件保留早期小样本冒烟协议（草堆 15）供对照，**不得**与规模化主表混用。

- 评测 / 评判模型：本地 Ollama `qwen3:8b`（关闭思维链）
- 嵌入模型：`nomic-embed-text-v2-moe`（768 维）
- 复用评测框架：`eval/eval_new.py::LongMemEvalEvaluator`
- 复现脚本：`eval/legal/prepare_legal_datasets.py`、`eval/legal/run_legal_eval.py`
- **规模化主实验**：`eval/legal/run_legal_scaled.py` → `REPORT_SCALED.md`
- **投稿前修订协议**（paraphrase/follow-up + 标准基线）：`eval/legal/run_revision_protocol.py` → `results/legal_revision/`

---

## 0. 与论文主表的对应关系（重要）

| 协议 | 规模 | 用途 | 文档 |
|------|------|------|------|
| 早期 per-instance needle（冒烟） | 草堆 15；可扩展至论文 P1 $n{=}120$ | P1 QA 构造 | 本文件 + `expand_prepare` |
| 共享语料主协议 | $M{=}400$, $S{=}300$ | 论文主表 EAR/EC | `REPORT_SCALED.md` |
| 修订查询协议 | 同上 store + exact/paraphrase/follow-up | 防 exact-replay 捷径 | `results/legal_revision/` |
| Dual-protocol QA | P1 $n{=}120$ + P2 $n{=}150$ → **$N{=}270$/库** | 端到端生成 | 论文 Table QA |

指标更名（与修改意见一致）：原报告中的 `session_recall@k` / 文稿旧称 S-R@k 对应 **Episode Completeness@k（EC）**；
另报告 **Session Hit@k** 与 **Answer Hit@k**。计分与论文 Eq.(2) 对齐为 **加法式** $\alpha S+(1-\alpha)R$（见 `memory_manager.py`）。

---

## 1. 实验设计

### 1.1 目标
论文在 **LongMemEval** 与 **LoCoMo** 两个通用长对话数据集上报告了两项核心指标：

| 指标 | 含义 |
|------|------|
| **检索召回率 RR（recall@k）** | 金标准证据会话的对话片段是否被检索进 Top-k |
| **QA 正确率** | 由 LLM 评判生成答案相对标准答案的正确性（A/B/C/D → 1.0/0.7/0.3/0.0）|

本实验把同一套指标迁移到**法律咨询**这一垂直领域，验证 BIMS 的"情景—语义"双记忆与自适应检索
在专业领域长对话中的稳定性与可迁移性。

### 1.2 难点：法律咨询数据不是长对话
权威法律咨询数据集通常是**单轮问答对**（用户问题 → 专家/律师解答），不具备 LongMemEval 所需的
**多会话长上下文**结构。因此需要对其做**数据预处理**，构造"大海捞针"(needle-in-haystack)长对话。

### 1.3 大海捞针构造法（与 LongMemEval 对齐）
对每个评测样本 `instance`：

1. 选定一条目标咨询问答对作为**"针"（证据会话）**：其问题作为 `question`，其专家答案作为金标准 `answer`。
2. 把"针"包装成一个 `[user 提问, assistant 解答]` 的双轮会话。
3. 随机采样 `haystack_size − 1` 条**同领域**法律咨询作为干扰会话（hard negatives），与"针"一起组成长对话草堆，"针"插入随机位置。
4. `answer_session_ids` 指向"针"所在会话——即检索召回率的金标准。

该结构与 `haystack_sessions / haystack_session_ids / answer_session_ids` 字段**逐一对应**，因此
可以**零改动复用** `LongMemEvalEvaluator` 的加载、检索评估、问答评估与评判全流程。

### 1.4 评测配置
- `haystack_size = 15`（每样本 15 个会话 × 2 轮 = 30 条对话记录，全部写入记忆系统）
- `sample_size = 12`（每个数据集 12 个评测样本）
- `max_retrieved_items (k) = 10`
- `seed = 42`（"针"选取与干扰采样可复现）
- 问题类型统一标记为 `legal-consultation`

---

## 2. 数据集

经 HuggingFace 检索（按下载量/点赞数筛选权威来源），选定两个**中文法律咨询**数据集。
`legal-advice-reddit` 等数据集因只含分类标签、无专家答案，不适配 QA 正确率指标，已排除。

| 数据集 | HF 仓库 | 来源 / 权威性 | 规模(咨询问答对) | 本实验取用子集 |
|--------|---------|----------------|------------------|----------------|
| **DISC-Law-SFT** | `ShengbinYue/DISC-Law-SFT` | 复旦大学 DISC-LawLLM（★176） | 79,692 | `legal_question_answering`（全部为法律问答）|
| **Lawyer-LLaMA** | `Skepsun/lawyer_llama_data` | 北大 Lawyer-LLaMA 项目 | 6,434 | `legal_advice` / `legal_counsel_with_article` / `legal_counsel_multi_turn` |

加载方式：`huggingface_hub.hf_hub_download` 下载原始文件至 `data/legal_raw/`，由
`prepare_legal_datasets.py` 解析。

---

## 3. 数据预处理与模型加载

### 3.1 预处理（`eval/legal/prepare_legal_datasets.py`）
- **抽取问答对**：DISC 取 `input/output`；Lawyer-LLaMA 取咨询子集的 `instruction/output`。
- **质量过滤**：去除空值；问题长度 5–300 字、答案 ≥ 25 字；按问题前缀去重；剔除 `<think>` 残留。
- **构造长对话**：按 1.3 的方法生成 LongMemEval 兼容样本，输出至
  `data/legal/<dataset>/longmemeval_oracle.json`。

```bash
python eval/legal/prepare_legal_datasets.py --sample_size 12 --haystack_size 15 --seed 42
```

### 3.2 模型加载
- **记忆系统**：`VectorMemoryManager()`，逐条 `add_dialog(role, text)` 写入草堆，内部完成
  嵌入（Ollama `nomic-embed-text-v2-moe`，768 维）、句/段向量入库、双阶段聚类与摘要。
- **生成 / 评判 LLM**：Ollama `qwen3:8b`。CPU 环境下 qwen3 默认思维链会使单次生成慢 30–50 倍，
  故通过 `think:false`（`NoThinkOllamaClient`）关闭，并限制 `num_predict`，保证可控时延。

---

## 4. 测试脚本（`eval/legal/run_legal_eval.py`）

复用 `LongMemEvalEvaluator`：每个样本先 `reset` 记忆、写入整段草堆并建立
`answer_session_ids → 片段 tid` 映射，然后 `search(question, top_k=10)` 检索，按金标准证据
计算 `recall@k / precision@k / ndcg@k`，再用检索结果拼接提示词调用 LLM 生成答案，最后由
同一 LLM 评判正确性。

```bash
python eval/legal/run_legal_eval.py \
    --datasets disc_law lawyer_llama \
    --model qwen3:8b --max_retrieved_items 10 --num_predict 384
```

结果落盘至 `results/legal/<dataset>/`（指标汇总 / 逐样本明细 / 可视化图），并汇总到
`results/legal/legal_eval_summary.json`。

---

## 5. 测试结果与分析

### 5.1 主结果（每数据集 12 个样本，k=10）

| 数据集 | 检索召回率 RR | precision@k | nDCG@k | QA 正确率 | 成功率 |
|--------|:-------------:|:-----------:|:------:|:---------:|:------:|
| **DISC-Law-SFT** | **0.792** | 0.159 | 0.811 | **0.800** | 1.000 |
| **Lawyer-LLaMA** | **0.917** | 0.187 | 0.905 | **0.950** | 1.000 |
| 平均 | 0.854 | 0.173 | 0.858 | 0.875 | 1.000 |

> QA 正确率分布：DISC-Law-SFT 为 {1.0×4, 0.7×8}（中位数 0.7，σ=0.14）；
> Lawyer-LLaMA 为 {1.0×10, 0.7×2}（中位数 1.0，σ=0.11）。两数据集均无 C/D（错误）判定。

### 5.2 与论文通用数据集对比

| 数据集（领域） | 检索召回率 RR | QA 正确率 |
|----------------|:-------------:|:---------:|
| LongMemEval（通用，论文 Table 2）| 0.595 | 0.707 |
| LoCoMo（通用，论文 Table 2）| 0.521 | 0.682 |
| **DISC-Law-SFT（法律，本测试）** | **0.792** | **0.800** |
| **Lawyer-LLaMA（法律，本测试）** | **0.917** | **0.950** |

> 说明：论文使用统一基座（如 GPT-OSS-20B）在更大规模、更长草堆上评测；本测试为本地 `qwen3:8b`、
> 12 样本、15 会话草堆的小规模复现。两者**指标定义一致但实验规模/模型不同，数值不可直接横向等同**，
> 仅用于说明 BIMS 在法律垂直领域同样有效，且检索/问答指标与通用领域处于可比量级（甚至更高）。

### 5.3 分析
- **检索（RR / nDCG）**：两数据集 nDCG@k 均 > 0.80，说明在 14 条**同领域**强干扰中，BIMS 仍能把
  证据会话排到 Top-k 靠前位置。Lawyer-LLaMA（0.917）高于 DISC（0.792），因为 DISC 法律问答更"教科书化"、
  问句间语义更相近，干扰更强；precision@k 较低（≈0.17）是正常现象——每个样本仅 2 条金标准片段、k=10，
  上限即 0.2。
- **问答（QA 正确率）**：检索命中后，BIMS 把证据答案注入上下文，`qwen3:8b` 多数能复述/归纳出正确解答。
  Lawyer-LLaMA 答案更口语、结论明确，故正确率更高（0.95）；DISC 答案常含大量法条细节，生成答案易"部分正确"（0.7）。
- **稳定性**：两数据集成功率均为 1.000，全部样本端到端跑通，无失败实例。

### 5.4 案例
- **DISC-Law-SFT**：问"人身安全保护令的申请流程?" → 检索 recall@k=1.0，生成答案完整复现申请要件，评判 **A（1.0）**。
- **Lawyer-LLaMA**：问"在 X 教育分期付款怀疑有问题想解除合约怎么办?" → 检索 recall@k=1.0，生成答案给出
  联系机构/查看条款/保留证据等步骤，评判 **A（1.0）**。

### 5.5 可视化
- `results/legal/disc_law/metrics_comparison_new_oracle_*.png`
- `results/legal/lawyer_llama/metrics_comparison_new_oracle_*.png`
- 各数据集 `heatmap_comparison_new_oracle_*.png`

### 5.6 局限与后续
- 受 CPU + `qwen3:8b` 算力限制，样本量（12/数据集）与草堆（15 会话）较小；增大后数值会更稳健。
- 评测 / 评判使用同一模型，存在自评偏置；论文用更强基座可缓解。
- `question` 与"针"中的用户问句一致，user 侧检索偏易；后续可引入 LLM 改写问句、或采用多轮咨询会话
  （Lawyer-LLaMA `legal_counsel_multi_turn`）构造更贴近真实的长对话，进一步提升难度。
- 可补充消融（`no_temporal / no_assoc / single_stage_cluster`）以分析各模块在法律领域的贡献。
