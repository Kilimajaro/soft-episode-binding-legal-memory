# IPM 修订问题清单核对（相对 `IPM稿件修订问题清单.docx`）

更新日期：2026-07-30（投稿级逐条修缮后）  
稿件：`paper/ipm/ipm-article.tex` / `ipm-article-anonymous.tex` / `ipm-titlepage.tex`

## 条目状态（被动回应，正文自洽）

| ID | 状态 | 正文落点 |
|----|------|----------|
| S0-1 β hinge 闭式解 | **PASS** | β 节改为 score-competition + validation sweep；零 margin hinge 仅一句说明无信息 |
| S0-2 feasible-band | **PASS** | Prop.\ 限定 soft-injection `$s_a\le\beta s_h$`，并区分已有高 sibling score |
| S0-3 nDCG 固定 IDCG | **PASS** | Protocol 明确 gold-session IDCG；主表 `tab:ndcg-graded`；单元测试 smoke 通过 |
| S0-4 时间衰减/算法一致 | **PASS** | 30 天线性衰减；Alg.\ SoftO2 = expand + completeness + top-$k$ |
| S1-1 joint/BM25-joint | **PASS（边界主张）** | 主文一次说清为更粗粒度设计；局限节留未来同协议比较 |
| S1-2 Soft O2-C / glue | **PASS（定位 B）** | Soft O2 主贡献；Mix 主表；同 session 附录；glue 为共现构造而非部署设定；Hybrid 仅 Lawyer 标显著 |
| S1-3 RQ1 taxonomy | **PASS** | `tab:rq1-failure`；RQ1 改为 prevalence 而非 dominate |
| S1-4 QA 因果 | **PASS** | 单系统审计；关联表述；不声称相对 FlatIP 降幻觉 |
| S1-5 β 校准通道 | **PASS（收缩）** | exact proxy 坦白；AH sweep 含 paraphrase；advice 同默认无单独 fit |
| S1-6 多重比较 | **PASS** | 预定义 Soft O2 vs FlatIP 主族 + Holm；Hybrid 次级/定向 |
| S2-1 session-aware RW | **PASS** | parent/hydration/session + 对照表 |
| S2-2 多 embedding | **LIMITATION** | 局限明确单编码器 |
| S2-3 多种子 | **LIMITATION** | 主表 seed 42；$M{=}400$ 五种子 |
| S2-4 一键复现 | **PASS（smoke）** | `bash scripts/reproduce_ipm_smoke.sh`；README 已改 |
| S2-5 数据许可 | **PASS（基本）** | `LICENSE`（MIT 代码）+ 上游原许可声明 |
| S3-1 Abstract ≤250 | **PASS** | ~176 词 |
| S3-2 双匿名 | **PASS** | 独立 anonymous 稿，无作者/仓库身份 |
| S3-3 Title page | **PASS** | 邮寄地址/邮编/致谢/CRediT；摘要回填 |
| S3-4 大空白页 | **PASS** | 去掉多余 clearpage；34 页无整页空白 |
| C-1 GenAI 写作声明 | **方法声明** | title page 仅披露 LLM-as-judge 研究方法 |

## 主文 / 附录表序

主文：`o1o2`, `rq1-failure`, `soft-o2-controls`, `ndcg-graded`, `ce`, `csce`, `qa_audit`  
附录：`cluster_same`, `sig`, `scale`, `beta`, grids, `bm25`
