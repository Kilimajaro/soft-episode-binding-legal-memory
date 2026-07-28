# BIMS-LEGAL Reproducible Code Package

Companion code for:

> **BIMS-LEGAL: Dual-Store Soft Binding for Recovering Prior Legal Advice under Same-Domain Interference**

**Authors:** Linrui Xu (`xu-l81@webmail.uwinnipeg.ca`); Linrui Han (corresponding, `linrui_han@163.com`)

Self-contained slice of the BIMS-LEGAL codebase: dual-store core, **O1** (FlatIP), **O2** (Soft O2 session binding), **O3** (bulk-load consolidation), **Soft O2-C** (cluster binding), fair Mix protocol, and manuscript table/figure scripts.

## Layout

```
BIMS-LEGAL-code/
├── memory_manager.py          # Dual-store BIMS core (Soft O2 / Soft O2-C)
├── embed_backend.py
├── config.py
├── requirements.txt
├── RESULTS_IPM.md             # Result index aligned with manuscript tables
├── eval/
│   ├── eval_new.py
│   ├── metrics.py
│   └── legal/                 # LegalEp / CAIL / scaled ablation scripts
│       └── v3/
│           ├── run_cluster_o2_ablation.py   # Soft O2 vs Soft O2-C / Hybrid
│           ├── build_split_episode_manifest.py  # fair Mix builder
│           └── run_legalmem_mt.py
├── paper/
│   ├── scripts/               # fill_v4_tables.py, fill_csce_tables.py, draw_*.py
│   └── ipm/figures/           # Generated figure assets (Fig1–Fig5)
└── scripts/
    ├── launch_csce_mix_fair.sh
    ├── launch_cluster_o2_ablation.sh
    └── launch_soft_o2_rerun.sh
```

## Environment

- Python 3.10+
- Ollama with `nomic-embed-text-v2-moe` (768-dim embeddings)
- `pip install -r requirements.txt`

Point data at the companion dataset package:

```bash
export BIMS_LEGAL_DATA=../BIMS-LEGAL-dataset
# In full repo layout, symlink into data/legal/:
# ln -s ../BIMS-LEGAL-dataset/csce_mix ../../data/legal/csce_mix
# ln -s ../BIMS-LEGAL-dataset/legalep_v4 ../../data/legal/legalep_v4
```

## Reproduction outline

1. **O1+O2 ablation** ($M{=}400$, $S{=}300$)
   ```bash
   python eval/legal/run_legal_scaled.py --help
   ```

2. **Soft O2 primary grids** (CAIL / LegalEp, $M{\approx}3000$)
   ```bash
   bash scripts/launch_soft_o2_rerun.sh
   python paper/scripts/fill_v4_tables.py
   ```

3. **Same-store cluster vs session** + **fair Mix Soft O2-C**
   ```bash
   bash scripts/launch_cluster_o2_ablation.sh
   SPLIT_RATIO=0.7 bash scripts/launch_csce_mix_fair.sh all
   python paper/scripts/fill_csce_tables.py
   ```

4. **Figures**
   ```bash
   python paper/scripts/draw_ipm_figures.py
   python paper/scripts/draw_fig1_pipeline.py
   ```

Primary numeric cells are mirrored under `../BIMS-LEGAL-dataset/primary_results/`.

## Notes

- Soft O2 is a retrieval-time binding policy over FlatIP; keep the encoder and turn index fixed when comparing operators.
- Exact replay is diagnostic only; primary claims use paraphrase, follow-up, and advice-recall channels.
- QA audit uses generator `qwen3:14b` and independent judge `qwen3:32b` ($N{=}270$ per corpus).
