# IPM最终投稿前问题清单落实（2026-08-05）

## 已完成（代码 / 数据 / 论文一致性）

| ID | 处理 |
|----|------|
| S0-1 | 主表/附录 AH·EC·nDCG·failure 统一来自 `corrected_metrics_*.json`（`regenerate_unified_tables.py`）；禁止 V4 AH + fresh nDCG 拼接 |
| S0-2 | 路线 A：全文生产路径改为 **target-k KMeans**；Birch 仅保留为未使用 helper 说明 |
| S0-3 | Methods 写明真实 Soft O2 阶段顺序（raw expand → boost/augment → session-first → fusion → completeness） |
| S0-4 | Hybrid `_cluster_direct_hits` 在 session expansion **之前**记录；`test_hybrid_gate.py`（AST 顺序 + 行为片段） |
| S0-5 | Holm 从 V4 `per_query_ah` 自动生成 + `test_holm_adjust.py`；产物 `holm_primary_family.json` |
| S0-6 | 四类 failure taxonomy（含 answer-only），`test_legal_metrics.py` 覆盖 |
| S1-2 | Soft O2-C / Mix 定位为 conditional mechanism；全部 Hybrid 行标 `†` exploratory（门控修复后需重跑） |
| S1-3 | 根目录为 SoT；`scripts/sync_canonical_code.sh` 同步到 `BIMS-LEGAL-code/` 并 diff 校验 |
| S1-4 | Soft O2 vs Hard 改为 score-copy envelope / attenuated binder 叙事（统一重建上 Hard 可领先 AH） |
| S1-6 | `beta_star_hinge` → `coverage_quantile_diagnostic` |
| S2-3 | smoke：`bash scripts/reproduce_ipm_smoke.sh`（sync + gate/Holm/metrics + KMeans API） |
| S2-4 | `DATA_LICENSES.md` |
| S3-1 | 匿名稿：无 `\ead`、无真实邮箱/GitHub URL；贡献声明匿名化（`make_anonymous_tex.py`） |

## 未完成 / 需作者或算力

| ID | 说明 |
|----|------|
| S0-4 Mix 重跑 | **已完成（2026-08-06）**：全量 Mix 重跑入库；Hybrid 取消 exploratory；详见 `MIX_GATEFIX_RERUN_IMPACT_2026-08-06.md` |
| S1-1 joint-episode 实测 | 范围主张已收紧；同协议 latency/index 实测未补 |
| S1-5 paired ΔAH CI | Methods 已改为实际报告范围（Holm family）；完整 paired CI 可后续补 |
| S2-1/S2-2 | 多 embedding / 多种子仍为增强项 |
| S2-5 run metadata | corrected JSON 仍缺完整 commit/lock；建议下次重跑写入 |
| A-1/A-2 | 需作者事实确认，未代答 |

## 验证（本轮）

- `bash scripts/reproduce_ipm_smoke.sh` 通过
- `python paper/scripts/regenerate_unified_tables.py` 已回写主稿表
- `tectonic` 编译 `ipm-article.pdf` 与 `ipm-article-anonymous.pdf` 成功
