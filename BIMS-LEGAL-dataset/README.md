# BIMS-LEGAL Dataset Package

Companion dataset for:

> **BIMS-LEGAL: Dual-Store Soft Binding for Recovering Prior Legal Advice under Same-Domain Interference**

**Authors:** Linrui Xu; Linrui Han (corresponding)

Evaluation corpora, fair Mix manifests, and primary `results.json` files backing the IPM manuscript. Upstream Hugging Face / CAIL raw dumps remain under their original licenses.

## Layout

| Path | Contents |
|------|----------|
| `cail2024/` | Processed CAIL2024 consultation dialogues |
| `legalep_v4/` | LegalEp-DISC / LegalEp-Lawyer (`needles`, `distractors`, `queries`, manifests) |
| `legalmem_mt/` | Multi-turn LegalMem manifests / gold dialogs |
| `csce_mix/` | Fair Mix manifests + queries (70% cross / 30% same-session gold) |
| `disc_law/`, `lawyer_llama/` | LongMemEval-style oracle JSON for QA checks |
| `primary_results/bims_legal_cluster_o2/` | Same-session Soft O2 vs Soft O2-C |
| `primary_results/bims_legal_csce_mix/` | Fair Mix Soft O2-C / Hybrid (RQ4) |
| `primary_results/legal_scaled_o1o2/` | O1+O2 ablation summary ($M{=}400$) |
| `primary_results/scale_curve.json` | Scale / IVFPQ curve (Fig5, Table scale) |
| `release_summaries/` | QA $N{=}270$, revision summaries, human audit JSON |

> Soft O2 main grids for the manuscript ($M{\approx}3000$) live under `../paper/ipm/figures/corrected_metrics_*.json` (with per-query Answer Hit). Do not treat legacy `bims_legal_v4` Soft O2 dumps as the Soft O2 metric source.

## Provenance

- **CAIL2024** consultation tracks (Challenge of AI in Law organizers)
- **DISC-Law-SFT:** Hugging Face `ShengbinYue/DISC-Law-SFT`
- **Lawyer-LLaMA:** Hugging Face `Skepsun/lawyer_llama_data`
- Rebuilt LegalEp episodes and Mix splits are derived research artifacts

## Primary tier

Manuscript main tables use tier **M ≈ 3000** sessions, seed **42**, Soft O2 \(\beta=0.98\), Mix \(\beta_c=0.95\). See `primary_results/` and companion code for reproduction commands.

## Companion code

See [`../BIMS-LEGAL-code/README.md`](../BIMS-LEGAL-code/README.md).
