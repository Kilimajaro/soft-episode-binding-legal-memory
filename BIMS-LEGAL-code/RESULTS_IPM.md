# IPM BIMS-LEGAL — result index

**Manuscript:** `paper/ipm/ipm-article.tex`  
**Publication repo:** https://github.com/Kilimajaro/soft-episode-binding-legal-memory  
**Rebuild packages:** `bash scripts/build_publication_packages.sh`

## O1+O2 ablation ($M{=}400$, $S{=}300$)

| Corpus | Baseline AH | +O1 FlatIP | +O1+O2 Soft O2 |
|--------|------------:|-----------:|---------------:|
| DISC-Law | 0.540 | 0.567 | **0.780** |
| Lawyer-LLaMA | 0.397 | 0.397 | **0.680** |

Source: `results/legal_scaled/scaled_ablation_summary.json` → `primary_results/legal_scaled_o1o2/`

## Soft O2 primary grids ($M{\approx}3000$)

| Channel | FlatIP | Soft O2 | Hard hydr. | Shuffled O2 |
|---------|-------:|--------:|-----------:|------------:|
| CAIL U1 | 0.570 | **0.788** | 0.802 | 0.452 |
| CAIL Uk | 0.270 | **0.698** | 0.720 | 0.230 |
| CAIL U-last | 0.245 | **0.762** | 0.790 | 0.200 |
| Lawyer exact | 0.692 | **0.752** | 0.612 | 0.570 |
| Lawyer para | 0.568 | **0.759** | 0.596 | 0.592 |
| Lawyer advice | 0.332 | **0.472** | 0.394 | 0.348 |
| DISC para | 0.547 | **0.648** | 0.592 | 0.469 |
| DISC advice | 0.342 | **0.428** | 0.402 | 0.310 |
| DISC exact | **0.640** | 0.638 | 0.614 | 0.448 |

Source: `results/bims_legal_v4/*/tier_M/results.json`

## Same-store Soft O2 vs Soft O2-C (AH@10)

| Channel | Soft O2 | Soft O2-C |
|---------|--------:|----------:|
| LegalMem U1 | **0.903** | 0.712 |
| LegalEp-DISC exact | **0.836** | 0.744 |
| LegalEp-Lawyer exact | **0.818** | 0.710 |

Source: `results/bims_legal_cluster_o2/*/tier_M/results.json`

## Fair Mix Soft O2-C / Hybrid (RQ4)

| Corpus | Soft O2 | Hybrid | mid-$p$ vs Soft O2 |
|--------|--------:|-------:|-------------------:|
| LegalEp-Lawyer Mix | 0.496 | **0.534** | 0.010 |
| LegalEp-DISC Mix | 0.482 | 0.486 | 0.81 |
| LegalMem-MT Mix | 0.618 | **0.646** | 0.18 |

Source: `results/bims_legal_csce_mix/*/tier_M/results.json`

## QA audit ($N{=}270$, generator 14b / judge 32b)

| Corpus | P1 ($n{=}120$) | P2 ($n{=}150$) | Pooled |
|--------|---------------:|---------------:|-------:|
| DISC-Law | 0.836 | 0.891 | 0.867 |
| Lawyer-LLaMA | 0.845 | 0.871 | 0.859 |

Source: `results/legal_qa_n270/`

## Manuscript scripts

- `paper/scripts/fill_v4_tables.py` — CAIL / LegalEp main tables
- `paper/scripts/fill_csce_tables.py` — Mix table (`tab:csce`)
- `paper/scripts/draw_ipm_figures.py` — Fig3–Fig5
- `paper/scripts/draw_fig1_pipeline.py` — Fig1 architecture
