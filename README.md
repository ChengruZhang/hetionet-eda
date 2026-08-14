# Hetionet EDA — Drug Repositioning Knowledge Graph Analysis

Exploratory data analysis of **Hetionet v1.0**, a biomedical knowledge graph connecting
drugs, genes, and diseases. The goal is to characterise graph structure, enumerate
metapath features, and establish a rigorous baseline for the
**Compound-treats-Disease (CtD)** link-prediction task (drug repositioning).

This EDA is designed to the standard of a professional data science project:
full English, no redundant visualisations, and analysis that directly informs
downstream modelling decisions.

---

## Notebooks

| Notebook | Contents |
|---|---|
| [`01_overview.ipynb`](eda/01_overview.ipynb) | Dataset validation, node/edge type distributions, class-imbalance analysis |
| [`02_structure.ipynb`](eda/02_structure.ipynb) | Degree distribution (power-law fit), connectivity, shortest paths, centrality |
| [`03_compound_disease.ipynb`](eda/03_compound_disease.ipynb) | CtD/CpD sparsity, metapath enumeration, DWPC feature baseline, OWA analysis, negative-sampling design |
| [`04_link_prediction.ipynb`](eda/04_link_prediction.ipynb) | Graph heuristics (Common Neighbors, Jaccard), spectral embeddings, logistic regression. **Writes the locked train/test split every later notebook reuses.** |
| [`05_link_prediction_method2.ipynb`](eda/05_link_prediction_method2.ipynb) | Node2Vec: biased random walks + Word2Vec, three pair operators (Hadamard / Concat / Cosine), bootstrap CIs |
| [`06_evaluation_deepdive.ipynb`](eda/06_evaluation_deepdive.ipynb) | Precision@K, score distributions, outliers in 64-D vs 2-D, where spectral and Node2Vec disagree, rank-average ensemble |
| [`07_graph_ablation.ipynb`](eda/07_graph_ablation.ipynb) | Four walk-graph variants — CtD-only, C∪G∪D, full Hetionet, and CtD plus weighted metapath shortcuts |
| [`08_gnn.ipynb`](eda/08_gnn.ipynb) | GCN and R-GCN link prediction, end-to-end supervised, sharing one training loop so the edge-type ablation is clean. Five-seed stability section shows single-seed AUPRC comparisons on this task are noise below ± 0.03 |
| [`09_sideeffect_rgcn.ipynb`](eda/09_sideeffect_rgcn.ipynb) | Does edge-type awareness stop the model confusing "treats" with "causes"? Adds Side Effect nodes and retrains both architectures |
| [`10_ensemble_v2.ipynb`](eda/10_ensemble_v2.ipynb) | Five methods combined: rank-average, best pair, weight-simplex search. **Pre-audit numbers** — nb11/nb12 later found the labels it tunes against are contaminated where the methods disagree most; see the status note inside |
| [`11_external_validation.ipynb`](eda/11_external_validation.ipynb) | ClinicalTrials.gov and Europe PMC validation, label-contamination audit, and the direction defect that motivates 09 and 12 |
| [`12_negative_sampling.ipynb`](eda/12_negative_sampling.ipynb) | Can better negatives replace the architectural fix? Hard negatives from Hetionet's own side-effect edges, plus registry-screened ones. Verdict confirmed across five seeds; **hard-25 is now the project's default sampler** |

Run them in order — each notebook depends on outputs described (but not re-computed) by the previous one.

Reading order note: 11 comes before 09 and 12 logically. It is the notebook that finds the problems those two attack, and it only needs 08's scores to run. The numbers stay as they are because renumbering breaks every cross-reference in the prose.


---

## Key Findings

| Finding | Value |
|---|---|
| Nodes / Edges | 47,031 / 2,250,197 |
| Node-type imbalance | 152.9× (Gene vs. Disease) |
| CtD density | 0.36% (755 edges) |
| Giant component (core subgraph) | 88% of nodes, mean path length 2.80 |
| CbG–GaD metapath CtD recall | ~73% with 36× coverage lift |
| UBC betweenness | 0.34 (next: 0.026) — hub correction mandatory |
| Common Neighbors baseline (no leakage) | AUROC 0.82 / AUPRC 0.030 (8× random) |
| Spectral embedding (L_sym, 4-dim) | AUROC 0.68 / AUPRC 0.005 — limited by heterogeneous graph structure |
| Node2Vec + Concat + LR | AUROC 0.921 / AUPRC 0.031 |
| **GCN, end-to-end supervised** | **AUROC 0.922 / AUPRC 0.179 / P@10 = 0.70 on the reference seed** — the jump is in the top of the ranking, not the average. Across 5 seeds: AUROC 0.915 ± 0.010, AUPRC 0.143 ± 0.031 — the 0.179 headline is the best draw, not the typical one |
| R-GCN (per-relation weights) | Below GCN on AUROC (4/5 seeds), statistically tied on AUPRC, ~40% slower. The earlier "GCN stable, R-GCN not" claim inverted under 5 seeds: GCN's AUPRC spread is twice R-GCN's |
| **Single-seed AUPRC gaps < ± 0.03 are seed noise** | Established by nb08 §8 and confirmed by nb12 §6 — method comparisons on this task need multi-seed or paired designs |
| Widening the walk graph past Gene | +0.01 AUROC for 27k extra nodes — Gene is where the signal stops |
| Gene as bridge, not as node (V4) | 99.8% of the C∪G∪D result using 8% of the nodes |
| Five-method ensemble | AUROC 0.958; spectral gets weight 0.0 once the GNNs are present. *Pre-audit labels — treat as an upper bound until re-run under screened negatives* |
| External validation (matched, N=300) | Phase 2+ enrichment 1.33× [1.10, 1.63]; post-2017 enrichment not significant |
| Literature validation (Europe PMC) | 69.7% studied vs 60.7% control; but "new since 2017" runs *below* control (0.63×) |
| **Contamination in `y = 0`** | **~12% overall, 28% in the top-scoring decile** — AUPRC is biased, not merely noisy |
| **The model confuses "treats" with "causes"** | 8.9% ± 1.8 of top-300 novel predictions are drugs known to *cause* the condition (5-seed mean; the old 6.0% was the luckiest seed). Edge-type awareness cuts it to 2.3%; a 25% hard-negative mix cuts it to 0.9% at no ranking cost |
| **Adopted: hard-25 negative sampler** | Hard negatives from Hetionet's own `causes` edges, 25% of each batch. causes@300 drops 10× on 5/5 seeds, AUPRC cost −0.004 ± 0.025 (noise). Beats the architectural fix at half the compute — now the project default |

---

## Setup

### 1. Get the data

Hetionet v1.0 is publicly available from the [Hetionet repository](https://github.com/hetio/hetionet).
Download the compressed JSON and place it at the path expected by `utils.py`:

```bash
# From the project root
mkdir -p hetionet-main/hetnet/json
curl -L "https://github.com/hetio/hetionet/raw/main/hetnet/json/hetionet-v1.0.json.bz2" \
     -o hetionet-main/hetnet/json/hetionet-v1.0.json.bz2
```

Alternatively, set the `HETIONET_PATH` environment variable to point to the file directly:

```bash
export HETIONET_PATH=/path/to/hetionet-v1.0.json.bz2
```

### 2. Install dependencies

This project uses [pixi](https://pixi.sh) for environment management.

```bash
# Install pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# From the eda/ directory
cd eda
pixi install
```

> **Other platforms**: `pixi.toml` targets `osx-arm64`. Linux/Windows users can install
> dependencies manually via pip:
> ```bash
> pip install jupyter pandas matplotlib seaborn networkx scipy scikit-learn tqdm
> ```

### 3. Launch Jupyter

```bash
cd eda
pixi run notebook
# or: jupyter notebook
```

---

## Project Structure

```
.
├── eda/
│   ├── 01_overview.ipynb              # run in order
│   ├── 02_structure.ipynb
│   ├── 03_compound_disease.ipynb
│   ├── 04_link_prediction.ipynb       # writes the locked split
│   ├── 05_link_prediction_method2.ipynb
│   ├── 06_evaluation_deepdive.ipynb
│   ├── 07_graph_ablation.ipynb
│   ├── 08_gnn.ipynb                   # GCN + R-GCN
│   ├── 09_sideeffect_rgcn.ipynb       # treats vs causes, architectural route
│   ├── 10_ensemble_v2.ipynb
│   ├── 11_external_validation.ipynb   # registry + literature + label audit
│   ├── 12_negative_sampling.ipynb     # treats vs causes, data route
│   ├── artifacts/
│   │   ├── splits/                    # locked pair universe and train/test indices
│   │   ├── predictions/               # per-method scores and *_meta.json receipts
│   │   └── external/                  # registry cache and validation outputs
│   ├── utils.py                       # shared data-loading and graph utilities
│   ├── pixi.toml                      # environment spec
│   └── pixi.lock
└── README.md
```

`artifacts/*_meta.json` files are checked in as reproducibility receipts; the large
score files they describe are gitignored and regenerate by re-running the notebook
that produced them. `artifacts/external/ctgov_pair_cache.pkl` is checked in as well —
it pins the ClinicalTrials.gov snapshot nb11's numbers were computed against, and
lets that notebook re-run without touching the registry.

### GPU note

nb08 and nb11 use PyTorch Geometric. `pixi install` covers `torch` and
`torch-geometric`, but `pyg-lib` (needed by `torch_geometric.nn.Node2Vec` in nb07)
is not on PyPI and has to come from the PyG wheel index:

```bash
pixi run uv pip install pyg-lib -f https://data.pyg.org/whl/torch-2.11.0+cpu.html
```

Training runs on MPS (Apple Silicon), CUDA, or CPU, selected automatically.

---

## References

- Himmelstein, D. S. et al. (2017). *Systematic integration of biomedical knowledge
  prioritizes drugs for repurposing.* eLife.
  [doi:10.7554/eLife.26726](https://doi.org/10.7554/eLife.26726)
- Hetionet repository: <https://github.com/hetio/hetionet>
