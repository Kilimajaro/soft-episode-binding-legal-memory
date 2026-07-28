# IPM legal revision — result index

**Tag:** `ipm-legal-revision-20260720`  
**Repo:** https://github.com/Kilimajaro/Vector-Memory-Is-All-You-Need

## Primary retrieval (paraphrase, $M{=}400$, $S{=}300$, $k{=}10$)

| Dataset | FlatIP AH | Soft O2 AH | Parent hydration | Shuffled O2 |
|---------|----------:|-----------:|-----------------:|------------:|
| DISC-Law-SFT | 0.747 | **0.800** | 0.613 | 0.577 |
| Lawyer-LLaMA | 0.703 | **0.750** | 0.420 | 0.617 |

Source: `results/legal_revision/*/revision_protocol.json`

## Dual-protocol QA ($N{=}270$, `qwen3:14b`)

| Dataset | P1 ($n{=}120$) | P2 ($n{=}150$) | Pooled |
|---------|---------------:|---------------:|-------:|
| DISC-Law-SFT | 0.836 | 0.891 | 0.867 |
| Lawyer-LLaMA | 0.845 | 0.871 | 0.859 |

Source: `results/legal_qa_n270/p1/legal_eval_summary.json`, `results/legal_qa_n270/p2/*/legal_full_ablation.json`

## Independent judge + human audit (manuscript tables)

Filled manuscript numbers (author declaration file separate):  
`results/legal_qa_n270/judge_human_audit_filled.json`

## Manuscript

- `paper/ipm/ipm-article.tex` / `paper/ipm/ipm-article.pdf`
- Figures: `paper/figures/fig2`–`fig9` (also under `paper/ipm/figures/`)
