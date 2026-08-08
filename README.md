# BIMS-LEGAL Publication Release

Companion release for the IPM manuscript:

> **BIMS-LEGAL: Dual-Store Soft Binding for Recovering Prior Legal Advice under Same-Domain Interference**

**Authors:** Linrui Xu (University of Winnipeg); Linrui Han (CUPL Data Law Lab / Institute for Data Law; corresponding author)

**Repository:** https://github.com/Kilimajaro/soft-episode-binding-legal-memory

## Quick start (smoke)

```bash
bash scripts/reproduce_ipm_smoke.sh
```

This syncs canonical code into `BIMS-LEGAL-code/`, runs Hybrid-gate / Holm / metrics unit tests, and checks the Soft O2-C KMeans API. Full Soft O2 grids require Ollama embeddings and are documented under `BIMS-LEGAL-code/README.md`.

## Reproduction tiers

| Tier | Command | What it does |
|------|---------|----------------|
| Smoke | `bash scripts/reproduce_ipm_smoke.sh` | Code sync, Hybrid-gate / Holm / metrics tests, Soft O2-C KMeans smoke |
| Table-only | `python paper/scripts/regenerate_unified_tables.py` | Rebuild Soft O2 grids / failure taxonomy / Holm from unified `corrected_metrics_*.json` (incl. `per_query_ah`) |
| Code sync | `bash scripts/sync_canonical_code.sh` | Root `memory_manager.py` / `eval/` → `BIMS-LEGAL-code/` (zero-diff check) |
| Full rerun | See `BIMS-LEGAL-code/README.md` | Requires Ollama embeddings; regenerates stores and rankings |

Canonical implementation: **repository root** (`memory_manager.py`, `eval/`). `BIMS-LEGAL-code/` is a publication mirror kept in sync by `scripts/sync_canonical_code.sh`.

Mix Soft O2-C / Hybrid entrypoints live under root `eval/legal/v3/` (`run_cluster_o2_ablation.py`, `build_split_episode_manifest.py`).

Data licences: see [`DATA_LICENSES.md`](DATA_LICENSES.md).

## Packages

| Folder | Purpose |
|--------|---------|
| [`BIMS-LEGAL-code/`](BIMS-LEGAL-code/) | Reproducible code: BIMS core, O1–O3, Soft O2 / Soft O2-C, eval scripts |
| [`BIMS-LEGAL-dataset/`](BIMS-LEGAL-dataset/) | Evaluation corpora, Mix manifests, and slim primary artifacts (Mix / Soft O2-C / scale / O1+O2) |
| [`paper/ipm/`](paper/ipm/) | Author manuscript, anonymous manuscript, figures, and unified Soft O2 metric JSON |

Rebuild publication folders from the development checkout:

```bash
bash scripts/build_publication_packages.sh
```

## Manuscript alignment

| Claim | Primary artifact |
|-------|------------------|
| O1+O2 ablation ($M{=}400$) | `BIMS-LEGAL-dataset/primary_results/legal_scaled_o1o2/` |
| Soft O2 on CAIL / LegalEp ($M{\approx}3000$) | `paper/ipm/figures/corrected_metrics_*.json` only (tables + Fig.~3/4); Holm / paired AH from same files via `holm_primary_family.json` + `paired_ah_primary_family.json` |
| β selection grid | Table A.4 numbers come from LegalMem-MT β sweeps under `results/.../beta_sweep` (selection/sensitivity; not interchangeable with unified Soft O2 grids) |
| Same-store Soft O2 vs Soft O2-C | `BIMS-LEGAL-dataset/primary_results/bims_legal_cluster_o2/` |
| Fair Mix Soft O2-C / Hybrid (post-gate) | `BIMS-LEGAL-dataset/primary_results/bims_legal_csce_mix/` + `csce_mix/` |
| QA audit ($N{=}270$) | `BIMS-LEGAL-dataset/release_summaries/qa/` |
| Scale curve | `BIMS-LEGAL-dataset/primary_results/scale_curve.json` |

Do **not** reintroduce legacy `bims_legal_v4` Soft O2 dumps: they conflict with the unified FlatIP-rebuild grids used in the manuscript.

## Licences

- **Code** in this repository: MIT (see `LICENSE`).
- **Upstream corpora** (CAIL2024, DISC-Law-SFT, Lawyer-LLaMA): remain under their original licences; see [`DATA_LICENSES.md`](DATA_LICENSES.md). Processed LegalEp/LegalMem artifacts are derived research releases.
