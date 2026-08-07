# IPM feedback acceptance (2026-08-07)

## S0-1 Figure 3 / Figure 4 vs main tables — DONE
- `paper/scripts/draw_ipm_figures.py` reads only `corrected_metrics_{cail,disc,lawyer}.json`.
- Removed all `bims_legal_v4` figure data paths.
- Regenerated `fig3_cail_main` / `fig4_legalep_main` (PDF+PNG) with value assertions.
- Author + anonymous PDFs rebuilt.

## S0-2 McNemar/Holm paired archive — IN PROGRESS
- Scripts updated to export/consume `per_query_ah` from the unified rebuild:
  - `paper/scripts/recompute_corrected_metrics.py`
  - `paper/scripts/regenerate_unified_tables.py`
  - `paper/scripts/merge_per_query_into_corrected.py`
- Live rebuild: tmux session `recompute-pq` rewriting lawyer/disc/cail with paired hits.
- After JSON completes: merge → `regenerate_unified_tables.py` → refresh Table A.2 / PDFs.
- Note: corrupt FAISS dumps on lawyer/cail were rebuilt; Soft O2 point estimates match the locked grids so far (e.g. Lawyer exact Soft O2 = 0.976).

## S0-3 beta=0.98 evidence — DONE
- Table A.4 and Methods/Conclusion label the beta sweep as a **separate early development campaign**, not the primary Soft O2 evaluation.
- Absolute AH values in Table A.4 are not comparable to the unified rebuild grids.

## Repo cleanup — DONE (staged)
- Dropped duplicate `data/primary_results` and mirrored raw dumps under `data/` (canonical data stays in `BIMS-LEGAL-dataset/`).
- Removed pre-gate Mix archive, demo `app.py` / `ablation_eval.py`, obsolete one-off paper scripts / launchers, author-only checklist markdowns.
