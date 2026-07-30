# IPM 修订问题清单核对（相对 `IPM稿件修订问题清单.docx`）

更新日期：2026-07-30  
稿件：`paper/ipm/ipm-article.tex`（main `4435748` 之后含本轮表格修复）

## S0 投稿阻断项

| ID | 要求 | 状态 | 说明 |
|----|------|------|------|
| S0-1 β hinge 闭式解 | 改推导或降级经验诊断 | **已完成** | Soft O2 β 节改为 diagnostic / validation sweep，不再声称闭式最优 |
| S0-2 feasible-band 假设 | 补 `s_a ≤ β s_h` 等 | **已完成** | 命题限定 soft-injected sibling |
| S0-3 nDCG 固定 gold IDCG | 重算 + 测试 | **已完成** | `legal_metrics.py` 修复；附录/主表 corrected nDCG；单元测试存在 |
| S0-4 时间衰减与代码一致 | 公式对齐实现 | **已完成** | 正文改为 30 天线性衰减 |

## S1 重大审稿风险

| ID | 要求 | 状态 | 说明 |
|----|------|------|------|
| S1-1 BM25-joint / joint episode | 正面比较或边界主张 | **按作者决定完成** | 已删除 BM25-joint 头对头；明确不同粒度未作 Soft O2 同协议对照；保留 BM25-turn |
| S1-2 Soft O2-C / gold glue | 定位为上界或补自然证据 | **已完成（定位 B）** | Mix 为 solvability bound；同 session 负对照；Hybrid 为有效性算子 |
| S1-3 RQ1 failure taxonomy | 直接失败类别表 | **已完成（本轮加强）** | FlatIP / Soft O2 / Hard / Shuffled 四分法 |
| S1-4 QA 因果 | 删因果或补配对基线 | **已完成（收缩）** | 单系统审计；不声称相对 FlatIP 降幻觉 |
| S1-5 β 校准通道 | paraphrase/advice 校准 | **部分完成** | 已降级诊断；跨通道 gap law 未全重跑 |
| S1-6 多重比较 | Holm 等 | **已完成** | `tab:sig` 含 Soft O2 vs FlatIP Holm；Hybrid 标 exploratory |

## S2 增强项

| ID | 要求 | 状态 |
|----|------|------|
| S2-1 session-aware 相关工作 | **基本完成** | parent/hydration/session 文献与对照表 |
| S2-2 多 embedding | **未做** | 增强项 |
| S2-3 多种子 | **部分** | 旧 M=400 五种子；主表仍 seed 42 |
| S2-4 一键复现 README | **部分** | 脚本在仓；入口仍可再收紧 |
| S2-5 数据许可/版本 | **部分** | Data availability 有；LICENSE 仍弱 |

## S3 合规

| ID | 要求 | 状态 |
|----|------|------|
| S3-1 Abstract ≤250 词 | **已完成** | ~200 词 |
| S3-2 双匿名稿 | **已完成** | `ipm-article-anonymous.tex` |
| S3-3 title page | **基本完成** | 需投稿前人工再核电话/邮编 |
| S3-4 大空白页 | **基本完成** | 致谢已迁；投稿前再翻 PDF |

## 本轮表格问题（用户指出）对应关系

| PDF 观感 | LaTeX label | 处理 |
|----------|-------------|------|
| Table4 只有 FlatIP | `tab:rq1-failure` | 扩成 FlatIP / Soft O2 / Hard / Shuffled |
| Table5 无下封口且对照弱 | `tab:ndcg-corrected` | 补 `\bottomrule`；加 Shuffled；主文新增 `tab:soft-o2-controls`（V4 四系统） |
| Table6 全是 O2-C < O2 | `tab:cluster_same` | 明确为**负对照**；**Mix `tab:csce` 前置**为 Soft O2-C/Hybrid 有效性主表 |

## 仍建议投稿前再做

1. S1-5：至少在一个 paraphrase/advice 通道上报告 β 敏感性与主表一致  
2. S2-4：README 单一入口命令实测  
3. 翻 PDF 确认 Table 编号与 caption 叙事一致（负对照 vs Mix 有效性）
