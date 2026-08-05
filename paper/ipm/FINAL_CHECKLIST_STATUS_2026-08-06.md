# IPM最终投稿前问题清单 — 终检状态（2026-08-06）

对照 `../IPM最终投稿前问题清单_2026-08-05.docx`。A-1/A-2 按作者要求不代答。

## S0 投稿阻断项

| ID | 状态 | 处理摘要 |
|----|------|----------|
| S0-1 | **PASS** | Soft O2 主表/附录 AH·EC·nDCG·failure 统一来自 `corrected_metrics_*.json`；已清除 LegalEp exact V4 残留（现用重建 0.924/0.976）；CE 表明确为 **secondary CE campaign**，不再与统一重建 Soft O2 数字混读 |
| S0-2 | **PASS** | 全文生产路径 target-k KMeans；Birch 仅 unused helper；代码注释已去 BIRCH 簇表述 |
| S0-3 | **PASS** | Methods + Algorithm~\ref{alg:soft-o2} 写明实现顺序：raw expand → boost/augment → session-first → fusion → completeness |
| S0-4 | **PASS** | 门控已修；Mix 全量重跑入库；`tab:csce` 与 `bims_legal_csce_mix` 一致；无 exploratory † |
| S0-5 | **PASS** | Holm 由 V4 `per_query_ah` 自动生成 + 单测；Methods 标明 paired 来源与重建点估计可略有数值差 |
| S0-6 | **PASS** | 四类 failure taxonomy；行和≈100%；单元测试覆盖 answer-only |

## S1 重大审稿风险

| ID | 状态 | 处理摘要 |
|----|------|----------|
| S1-1 | **PARTIAL（范围收紧）** | joint-episode 明确为不同粒度/未来工作；主表不含 BM25-joint；摘要不以 episode-level 最优自居 |
| S1-2 | **PASS** | Soft O2-C/Mix 为 conditional；Hybrid 语料条件化（LegalEp 显著，LegalMem-MT Soft O2 仍优） |
| S1-3 | **PASS** | 根目录 SoT；`sync_canonical_code.sh`；Mix runners 已提升到 root `eval/legal/v3/` |
| S1-4 | **PASS** | Soft O2 = attenuated binder；Hard = score-copy envelope；结论/脚注不再声称 Soft O2 EC 全面优于 Hard |
| S1-5 | **PASS（按实际报告）** | Methods 声明 Holm family；附录去掉空的 95% CI 列 |
| S1-6 | **PASS** | 脚本/JSON 改为 `coverage_quantile_*`；明确非 optimizer |

## S2 复现与稳健性

| ID | 状态 | 处理摘要 |
|----|------|----------|
| S2-1 | **PARTIAL** | Limitations 披露单 encoder；未补多 encoder（算力增强项） |
| S2-2 | **PARTIAL** | 披露主表单 seed；五 seed 仅 M=400 |
| S2-3 | **PASS** | smoke + pytest 依赖 + table-only 再生命令 |
| S2-4 | **PASS** | `DATA_LICENSES.md` |
| S2-5 | **PARTIAL** | `corrected_metrics_*.json` 已加 provenance 注释字段；完整 lock 仍建议下次重跑写入 |

## S3 排版 / 双盲

| ID | 状态 | 处理摘要 |
|----|------|----------|
| S3-1 | **PASS** | 匿名稿无 `\ead`/邮箱/GitHub；贡献声明匿名化 |
| S3-2 | **作者操作** | competing interests 需按 Elsevier 投稿系统单独上传 |

## 验证命令

```bash
bash scripts/reproduce_ipm_smoke.sh
python paper/scripts/regenerate_unified_tables.py
python paper/scripts/make_anonymous_tex.py
tectonic -X compile paper/ipm/ipm-article.tex --outdir paper/ipm
tectonic -X compile paper/ipm/ipm-article-anonymous.tex --outdir paper/ipm
```

## 主结论（投稿叙事）

1. **主贡献 Soft O2**：相对 FlatIP 显著提升 AH/EC/nDCG、降低 incompleteness；Hard 为 score-copy envelope。  
2. **Soft O2-C / Hybrid**：条件机制；门控 Hybrid 在 LegalEp Mix 显著，LegalMem-MT Mix 上 Soft O2 仍最优。  
3. **实现与论文一致**：KMeans、Soft O2 阶段序、Hybrid direct-dense gate、统一指标源。
