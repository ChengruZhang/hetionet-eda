# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Exploratory data analysis of **Hetionet v1.0** — a biomedical knowledge graph (47,031 nodes / 2,250,197 edges) — building toward a link-prediction baseline for the **Compound-treats-Disease (CtD)** drug-repositioning task. The deliverable is a sequence of analysis notebooks, not a library or service.

## Environment & commands

Environment is managed by **pixi** (conda + pypi). All commands are run from the `eda/` directory.

```bash
cd eda
pixi install            # first-time setup
pixi run python -m ipykernel install --user --name hetionet-eda --display-name "hetionet-eda"
pixi run notebook       # launches jupyter on port 8889, no browser
```

The notebooks pin `kernel_name: hetionet-eda` in their metadata. Without the `ipykernel install` step above, `jupyter nbconvert --execute` and "Run All" will fail with `NoSuchKernel: hetionet-eda` — this is not in the README. For one-off batch execution without registering, override with `--ExecutePreprocessor.kernel_name=python3`.

`pixi.toml` declares three platforms (`osx-arm64`, `linux-64`, `win-64`) but `pixi.lock` was generated on osx-arm64; on Linux/Windows fall back to pip-installing the deps listed in the README.

There is **no test suite, linter, or build step**. Quality control is done by re-running notebooks top-to-bottom.

## Data

The raw dataset is **not committed** (see `eda/.gitignore`). `utils.py` expects it at `<repo>/hetionet-main/hetnet/json/hetionet-v1.0.json.bz2`, or wherever `$HETIONET_PATH` points. If the file is missing, every notebook will fail at the `load_hetnet()` call in cell 1 — check this first before debugging anything else.

Download command is in the README ("Setup → Get the data").

## Architecture

The four notebooks form a **strict linear pipeline**; each one builds on conclusions (not just code) from the previous. Re-running out of order produces inconsistent narrative even if cells execute:

1. `01_overview.ipynb` — dataset validation, node/edge-type distributions, class imbalance
2. `02_structure.ipynb` — degree distribution, connectivity, shortest paths, centrality (operates on the **core induced subgraph** of biologically relevant node kinds, not the full 47k graph, for tractability)
3. `03_compound_disease.ipynb` — CtD/CpD sparsity, metapath enumeration, DWPC features, negative-sampling design under the open-world assumption (OWA)
4. `04_link_prediction.ipynb` — Common Neighbors / Jaccard heuristics, spectral embeddings, logistic regression; evaluated with AUROC + AUPRC (AUPRC is the meaningful one given 0.36% CtD density)

`utils.py` is the only shared module. Its surface:
- `load_hetnet()` — reads the bz2 JSON
- `to_networkx()` — converts to a `MultiDiGraph` where node IDs are `(kind, identifier)` tuples; edges carry `kind` and `direction`
- `build_subgraph(G, kinds)` — induced subgraph by node kind; used to bound centrality cost
- `metanode_counts` / `metaedge_counts` / `degree_table` — descriptive helpers
- `setup_plot_style()` — Unicode-safe matplotlib + seaborn whitegrid theme; call once per notebook

Node identity convention: every node is keyed by `(kind, identifier)` end-to-end. Do not collapse to `identifier` alone — identifiers are only unique within a kind.

## Project conventions (from README)

- **English only** in notebook prose and plot labels; no redundant/decorative visualisations.
- Analysis should "directly inform downstream modelling decisions" — when adding a cell, the bar is that it changes a later choice (feature, sampling strategy, metric), not that it looks interesting.
- Hub correction is mandatory in any centrality- or path-based feature (UBC betweenness is 0.34 vs. 0.026 for the next node — leaving it uncorrected leaks structure).
- Link-prediction splits must be leakage-free; the headline CN baseline (AUROC 0.82 / AUPRC 0.030) was measured under that constraint and is the number new work is compared against.
- **Split is locked, not re-rolled.** nb4's tail persists the LR pair universe + train/test indices + baseline scores under `eda/artifacts/`. Any new method (nb5+) must `load` these and compute on the same `idx_train`/`idx_test`. Never call `train_test_split` again downstream. `artifacts/splits/split_meta.json` is the receipt and is checked into git; the data files are gitignored and regenerate by re-running nb4.
