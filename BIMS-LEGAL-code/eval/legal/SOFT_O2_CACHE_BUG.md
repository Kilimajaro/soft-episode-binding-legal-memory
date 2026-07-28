# Soft O2 search-cache bug（V4 / CAIL）

## 现象

在 `results/bims_legal_v4/` 的多份结果里，`dense_flat`、`dense_o2`、`shuffled_o2` 的
`per_query_ah` 与 AH/EC/nDCG/MRR **逐条完全相同**，且 Soft O2 / shuffled 的
`elapsed_seconds ≈ 0`，而 FlatIP 需上千秒。`parent_hydrate` / `session_max` 仍有大幅 AH
提升（例如 CAIL `u_last`：0.245 → ~0.79），说明 session 映射本身可用。

这**不是** Soft O2 在科学上等价于 FlatIP，而是评测结果被检索 LRU 缓存污染。

## 根因

1. `VectorMemoryManager.search()` 把排序结果缓存在与 embedding 共用的 `lru_cache` 中，
   旧缓存键只含 `query` + `is_temporal_task`，**不含** `_session_expand` /
   `_session_first_rerank` / `_session_coherence(β)`。
2. V4 runner `eval/legal/v3/run_legalmem_mt.py::apply_cfg` 为保留 embedding LRU，
   切换配置时**没有**清理 search 缓存（旧 `run_revision_protocol.py` 会
   `lru_cache.clear()`）。
3. 评测顺序为 `dense_flat → dense_o2 → shuffled_o2`，FlatIP 写入缓存后，后续 Soft O2 /
   shuffled / β 全部命中 FlatIP 排序。

对照：`legalep_disc_M` 是少数 Soft O2 真正重算过的（`elapsed≈1551s`，shuffled AH 明显下降）。

## 修复（2026-07-27）

1. **cache key**：`search()` 键加入 `top_k`、expand、session-first、β、`use_pq`。
2. **切配置清缓存**：`apply_cfg` 调用 `clear_search_cache()`（只删 `search:` 前缀，保留
   `emb:` embedding LRU）。
3. **β 通道**：新增 `--beta_channel`（CAIL β 扫 `uk_followup`）。

## 补跑

```bash
bash scripts/launch_soft_o2_rerun.sh
```

| GPU | 任务 | 新鲜输出 | 合并回 |
|-----|------|----------|--------|
| 0 | CAIL `dense_o2`+`shuffled_o2`+β(`uk_followup`) | `cail_o2fix/` | `cail_M`（并刷新 `cail_ce`/`cail_beta` 的 O2 单元） |
| 1 | Lawyer exact / Lawyer para / DISC para 同上 | `legalep_*_o2fix/` | `legalep_lawyer_M`、`legalep_*_para`、对应 beta |

日志：`results/bims_legal_v4/logs/soft_o2_rerun.log` 及 `*_o2fix.log`。

合并后的 `results.json` 会带 `"soft_o2_cache_fix"` 字段标记补丁来源。
**在补跑 `ALL_DONE` 之前，勿把表中 Soft O2 / shuffled / β 当作有效结论。**

可信且未受该 bug 污染的单元（可继续引用）：BM25、CE rerank、`parent_hydrate`、
`session_max`、以及 `legalep_disc_M` 中已真实重算的 Soft O2。

本说明入库路径：`eval/legal/SOFT_O2_CACHE_BUG.md`（`results/` 目录被 gitignore，本地可另存副本）。

## 补跑进度（启动后即时观测）

`tmux` 会话：`soft-o2-rerun`（`bash scripts/launch_soft_o2_rerun.sh`）。

启动后 Soft O2 已在真实重算（不再 `elapsed≈0`）。相对旧 FlatIP 基线的中途读数示例：

| 任务 | 旧 FlatIP AH（可信） | Soft O2 中途 AH（补跑中） |
|------|---------------------:|--------------------------:|
| Lawyer exact | 0.692 | ~0.83（250/500） |
| CAIL `u1_exact` | 0.570 | ~0.74（50/600） |

以日志 `ALL_DONE` 与合并后的 `results.json`（含 `soft_o2_cache_fix`）为准。


## Advice Soft O2（2026-07-27 干净补跑）

`scripts/launch_advice_o2_rerun.sh` 双卡重跑 `advice_recall` 的 Soft O2 / shuffled，
并 merge 回 `legalep_*_advice/tier_M/results.json`：

| Corpus | FlatIP | Soft O2 | Hard hydr. | Shuffled |
|--------|-------:|--------:|-----------:|---------:|
| DISC advice | 0.342 | **0.428** | 0.402 | 0.310 |
| Lawyer advice | 0.332 | **0.472** | 0.394 | 0.348 |

手稿不再保留污染单元格或 “excluded / pending” 表述。
