# IPM 修订问题清单核对（相对 `IPM稿件修订问题清单.docx`）

更新日期：2026-07-30  
稿件：`paper/ipm/ipm-article.tex`（`6956e51`，投稿向叙事整理后）

## 主文 / 附录表布局（投稿版）

**主文：** `tab:o1o2`, `tab:rq1-failure`, `tab:soft-o2-controls`, `tab:ndcg-graded`, `tab:ce`, `tab:csce`（Mix 有效性）, `tab:qa_audit` + 主图  

**附录（靠前为叙事辅助）：** `tab:cluster_same`（同 session Soft O2 默认）, `tab:sig`, `tab:scale`, `tab:beta`, 然后 CAIL/DISC/Lawyer grids, `tab:bm25`

## S0 投稿阻断项

| ID | 要求 | 状态 | 说明 |
|----|------|------|------|
| S0-1 β hinge 闭式解 | 改推导或降级经验诊断 | **已完成** | Soft O2 β 节为 diagnostic / validation sweep |
| S0-2 feasible-band 假设 | 补 `s_a ≤ β s_h` 等 | **已完成** | 命题限定 soft-injected sibling |
| S0-3 nDCG 固定 gold IDCG | 重算 + 测试 | **已完成** | 主文 `tab:ndcg-graded`；附录网格含 nDCG |
| S0-4 时间衰减与代码一致 | 公式对齐实现 | **已完成** | 正文 30 天线性衰减 |

## S1 重大审稿风险

| ID | 要求 | 状态 | 说明 |
|----|------|------|------|
| S1-1 BM25-joint / joint episode | 正面比较或边界主张 | **已完成** | 不同粒度；不作为 Soft O2 同协议对照；保留 BM25-turn |
| S1-2 Soft O2-C / gold glue | 定位清晰 | **已完成** | 主文 Mix/`tab:csce`；同 session 对照在附录；glue 写为评价构造而非部署设定 |
| S1-3 RQ1 failure taxonomy | 直接失败类别表 | **已完成** | FlatIP / Soft O2 / Hard / Shuffled |
| S1-4 QA 因果 | 删因果或补配对基线 | **已完成** | 单系统审计；不声称相对 FlatIP 降幻觉 |
| S1-5 β 校准通道 | paraphrase/advice 校准 | **部分完成** | 已降级诊断；跨通道 gap law 未全重跑 |
| S1-6 多重比较 | Holm 等 | **已完成** | `tab:sig`（附录靠前）含 Holm |

## S2 / S3

| ID | 状态 |
|----|------|
| S2-1 session-aware 相关工作 | **基本完成** |
| S2-2 多 embedding | **未做** |
| S2-3 多种子 | **部分**（旧 M=400 五种子；主表 seed 42） |
| S2-4 一键复现 README | **部分** |
| S2-5 数据许可/版本 | **部分** |
| S3-1 Abstract ≤250 词 | **已完成** |
| S3-2 双匿名稿 | **已完成**（与主稿同步，无 revised/corrected 痕迹） |
| S3-3 title page | **基本完成** |
| S3-4 大空白页 | **基本完成** |

## 投稿前可选增强

1. S1-5：至少一个 paraphrase/advice 通道报告 β 敏感性与主表一致  
2. S2-4：README 单一入口命令实测  
3. 人工翻 PDF 确认匿名稿作者信息已剥离、表序与正文引用一致
