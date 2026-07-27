# Soft Episode Binding for Legal Consultation Memory

Reproduction package for the paper:

**Soft Episode Binding for Consultation Memory Retrieval: Recovering Prior Legal Advice under Same-Domain Interference**

This repository provides the code, evaluation corpora, and primary result files used in the manuscript.

## Contents

| Path | Description |
|------|-------------|
| `memory_manager.py` | BIMS dual-store memory and Soft O2 scoring |
| `eval/legal/` | LegalEp / CAIL preparation and evaluation scripts |
| `data/` | Processed CAIL, LegalEp, oracle JSON, and primary `results.json` cells |
| `paper/scripts/` | Table/figure regeneration helpers |
| `requirements.txt` | Python dependencies |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Embeddings and optional generation use a local [Ollama](https://ollama.com) server. Default model names are set in `config.py`.

## Reproduce (outline)

1. Point data paths to `data/` (or symlink as `../data/legal` from a full checkout layout).
2. Run Soft O2 / FlatIP / hard-expansion / BM25 / CE controls via `eval/legal/` and `scripts/`.
3. Regenerate manuscript tables/figures with `paper/scripts/fill_v4_tables.py` and `paper/scripts/draw_ipm_figures.py`.

Upstream corpora remain under their original licenses (CAIL2024; Hugging Face `ShengbinYue/DISC-Law-SFT`, `Skepsun/lawyer_llama_data`).

## Citation

Please cite the paper when using this code or data.
