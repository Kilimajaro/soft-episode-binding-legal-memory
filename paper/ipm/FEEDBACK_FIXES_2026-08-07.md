# IPM feedback acceptance (2026-08-07)

## S0-1 Figure 3 / Figure 4 vs main tables — DONE
- `paper/scripts/draw_ipm_figures.py` reads only `corrected_metrics_{cail,disc,lawyer}.json`.
- Removed all `bims_legal_v4` figure data paths.
- Regenerated `fig3_cail_main` / `fig4_legalep_main` after unified rebuilds.

## S0-2 McNemar/Holm paired archive — IN PROGRESS
- Export/consume `per_query_ah` from the unified rebuild.
- DISC: reuse intact `results/bims_recompute/disc_full` index so table AH stay locked.
- Lawyer/CAIL: rebuild after truncated indexes, then `--force` merge so Holm ΔAH equals Soft O2−FlatIP.
- Finish watcher: tmux `finish-s02` / `/tmp/finish_s02.sh`.

## S0-3 beta=0.98 evidence — DONE
- Table A.4 and Methods/Conclusion label the beta sweep as a **separate early development campaign**.

## Repo cleanup — DONE
- Dropped legacy `BIMS-LEGAL-dataset/primary_results/bims_legal_v4/` Soft O2 dumps.
- Removed one-off `launch_*.sh`, duplicate `data/` mirrors, and obsolete V4-sync paper scripts.
- Soft O2 manuscript numbers come only from `paper/ipm/figures/corrected_metrics_*.json`.
