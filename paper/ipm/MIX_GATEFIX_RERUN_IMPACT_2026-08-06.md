# Mix Soft O2-C / Hybrid 门控修复后重跑：对原结论与结构的影响

**Run:** 2026-08-06 02:08–03:44 CST  
**Code:** gate-fixed `memory_manager.py` (`_cluster_direct_hits` before session expand); `hybrid_xsess` with `_cluster_trigger_on_soft=False`  
**Protocol:** `SPLIT_RATIO=0.7`, β_c=0.95, cluster_max_siblings=16, configs `ep_flat sess_o2 cluster_o2 hybrid_xsess`  
**Artifacts:**
- New primary: `BIMS-LEGAL-dataset/primary_results/bims_legal_csce_mix/*/tier_M/results.json`
- Pre-gate archive: `BIMS-LEGAL-dataset/primary_results/bims_legal_csce_mix_pre_gatefix_2026-07-28/`
- Run logs/copy: `results/bims_legal_csce_mix_gatefix/`

## 数字对比（AH / AH_cross / AH_same）

| Corpus | System | OLD AH | NEW AH | OLD cross | NEW cross | mid-p vs Soft O2 (NEW) |
|--------|--------|--------|--------|-----------|-----------|-------------------------|
| LegalEp-DISC | FlatIP | 0.478 | 0.494 | 0.477 | 0.491 | — |
| | Soft O2 | 0.482 | 0.492 | 0.443 | 0.460 | — |
| | Soft O2-C | 0.410 | 0.498 | 0.443 | 0.506 | 0.73 (ns) |
| | **Hybrid** | **0.486** | **0.542** | **0.469** | **0.526** | **0.0005**** |
| LegalEp-Lawyer | FlatIP | 0.462 | 0.460 | 0.454 | 0.451 | — |
| | Soft O2 | 0.496 | 0.492 | 0.446 | 0.440 | — |
| | Soft O2-C | 0.464 | 0.464 | 0.491 | 0.480 | 0.13 (ns) |
| | **Hybrid** | **0.534** | **0.536** | **0.523** | **0.520** | **0.0022**** |
| LegalMem-MT | FlatIP | 0.534 | 0.530 | 0.526 | **0.520** | — |
| | **Soft O2** | **0.618** | **0.602** | 0.503 | 0.489 | — |
| | Soft O2-C | 0.500 | 0.440 | 0.537 | 0.460 | ≪0.01 **(worse)** |
| | Hybrid | **0.646** | 0.562 | **0.551** | 0.446 | 0.059 (ns) |

Glue/cluster diagnostics also moved (re-index under current KMeans path): DISC clusters 484→708; Lawyer 512→521; LegalMem-MT 903→635. So this is a **full Mix rebuild**, not a Hybrid-only delta on a frozen store.

## 对原结论的影响

### 不变 / 加强
1. **主结论 Soft O2 ≫ FlatIP（同会话 episode incompleteness）不受影响**——主表 Soft O2 grids、Holm family、failure taxonomy 未重跑，叙事仍以 Soft O2 vs FlatIP 为主。
2. **同会话仍优先 Soft O2**：附录 `tab:cluster_same` 未改；正文继续写 “gold 已共享 sid 时 Soft O2 足够”。
3. **门控 Hybrid 在 LegalEp Mix 上从 exploratory 升格为可报告显著性结果**：DISC/Lawyer 上 Hybrid 显著优于 Soft O2（**），且领跑 AH 与 AH_cross。这比预投稿 “Hybrid† 仅作历史 exploratory” 更强，也更符合 Algorithm Hybrid 的设计意图。

### 削弱 / 需收紧的原表述
1. **原 “Soft O2-C 在 cross 层抬升（Lawyer 0.491、LegalMem-MT 0.537）” 不再普遍成立**  
   - Lawyer cross 仍略高于 Soft O2（0.480 vs 0.440），但不再接近显著。  
   - LegalMem-MT cross：Soft O2-C **0.460 < Soft O2 0.489**，且 FlatIP 反超 cross（0.520）。  
   → Soft O2-C 应继续定位为 **conditional mechanism**，不能写成 Mix 上跨库稳定 cross 增益。
2. **原 LegalMem-MT Hybrid 最强（0.646）被推翻**：新 Hybrid 0.562 **低于** Soft O2 0.602。  
   → “gated Hybrid 一律最好” 不成立；**语料条件化**（LegalEp 有效，CAIL multi-turn Mix 无效甚至有害）。
3. **原 Soft O2-C 在 DISC 上显著差于 Soft O2（0.410**）消失**：新 Soft O2-C 0.498 ≈ Soft O2 0.492（ns）。门控+重聚类后 ungated Soft O2-C 的 same-session 伤害减轻，但 Hybrid 仍明显更好。

### 结构影响（论文章节）
| 部分 | 变化 |
|------|------|
| Abstract | 轻微：Soft O2-C 仍为 conditional；补一句 gated Hybrid 作为 composed operator（结构不变） |
| Intro / Methods | **结构不变**；Hybrid 门控描述已与代码一致，无需大改 |
| § Soft O2-C Mix (`sec:exp-cluster`) | **主要修订区**：换 `tab:csce` 全文数字；删 exploratory †；改写讨论为 LegalEp 上 Hybrid 显著、LegalMem-MT 上 Soft O2 仍最优 |
| Findings / Conclusion / Limitations | 对应收紧 Mix 表述；Holm 主族与 Soft O2 主实验结构不变 |
| Appendix same-session Soft O2-C | **不动** |
| 主 Soft O2 网格 / CE / scale | **不动** |

## 一句话结论

> **论文骨架与 Soft O2 主结论不变；唯一大改是 Mix 子实验：门控 Hybrid 在 LegalEp 上可正式报告并显著优于 Soft O2，但在 LegalMem-MT 上不再领先——Soft O2-C/Hybrid 必须保持语料条件化叙事，不能升格为全局主算法。**
