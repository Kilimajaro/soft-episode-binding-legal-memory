# IPM legal revision — result index

**Tag:** `ipm-legal-revision-20260720`  
**Repo:** https://github.com/Kilimajaro/Vector-Memory-Is-All-You-Need  
**Bundled JSON:** `release/ipm-legal-revision-20260720/`

## Soft O2 V4 (cache-fixed; primary manuscript numbers)

Search-cache collision (Soft O2 ≡ FlatIP) was fixed and re-run; primary
`results/bims_legal_v4/*/tier_M/results.json` cells carry
`soft_o2_cache_fix` where patched. Details:
`eval/legal/SOFT_O2_CACHE_BUG.md`.

Manuscript tables/figures: `paper/scripts/fill_v4_tables.py`,
`paper/scripts/draw_ipm_figures.py`.

### Soft O2 AH@10 (primary, $M{=}3000$)

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

Hard hydration and session-max aggregation coincide on all reported channels;
manuscript tables show a single hard-expansion baseline.

## Dual-protocol QA ($N{=}270$, `qwen3:14b`)

| Dataset | P1 ($n{=}120$) | P2 ($n{=}150$) | Pooled |
|---------|---------------:|---------------:|-------:|
| DISC-Law-SFT | 0.836 | 0.891 | 0.867 |
| Lawyer-LLaMA | 0.845 | 0.871 | 0.859 |

Source: `results/legal_qa_n270/p1/legal_eval_summary.json`, `results/legal_qa_n270/p2/*/legal_full_ablation.json`

## Manuscript

- `paper/ipm/ipm-article.tex` / `paper/ipm/ipm-article.pdf`
- Figures: `paper/ipm/figures/fig3`–`fig5`
