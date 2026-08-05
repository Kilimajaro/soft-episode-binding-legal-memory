# Data sources, licences, and redistribution notes

This repository mixes **MIT-licensed code** with **upstream legal corpora** that remain under their original terms. The MIT `LICENSE` at the repository root covers software only.

## Upstream corpora

| Corpus | Role in paper | Upstream source | Notes / redistribution |
|--------|---------------|-----------------|------------------------|
| CAIL2024 consultation tracks | Multi-turn Soft O2 grids / LegalMem-MT | CAIL2024 competition / organizers' release channels | Use under CAIL terms; do not treat bundled copies as redistributable beyond research evaluation without checking the organizer licence. |
| DISC-Law-SFT | LegalEp-DISC rebuild | Hugging Face `ShengbinYue/DISC-Law-SFT` | Upstream licence on the HF card applies. Filtered LegalEp artifacts are derived research releases. |
| Lawyer-LLaMA data | LegalEp-Lawyer rebuild | Hugging Face `Skepsun/lawyer_llama_data` | Upstream licence on the HF card applies. Filtered LegalEp artifacts are derived research releases. |

## Derived artifacts released here

| Artifact | Path | Contents |
|----------|------|----------|
| LegalEp / LegalMem manifests | `BIMS-LEGAL-dataset/` | Filtered session ids, Mix manifests, primary `results.json` |
| Unified Soft O2 metric rebuild | `paper/ipm/figures/corrected_metrics_*.json` | AH/EC/nDCG/failure taxonomy from one FlatIP rebuild |
| Holm primary family | `paper/ipm/figures/holm_primary_family.json` | Soft O2 vs FlatIP mid-$p$ + Holm |

Processing scripts live under `eval/legal/` and `paper/scripts/`. If an upstream licence forbids redistributing raw dumps, delete the corresponding raw files and retain only download/filter scripts plus aggregate metrics.

## Removal

To drop a corpus copy from a local checkout, remove the matching tree under `BIMS-LEGAL-dataset/` / `data/` and re-run download/build scripts documented in `BIMS-LEGAL-dataset/README.md` (when present) or `BIMS-LEGAL-code/README.md`.
