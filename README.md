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
| [`04_link_prediction.ipynb`](eda/04_link_prediction.ipynb) | Graph heuristics (Common Neighbors, Jaccard), spectral node embeddings, logistic regression baseline, AUROC/AUPRC evaluation |

Run them in order — each notebook depends on outputs described (but not re-computed) by the previous one.

### Tutorials

Bilingual (Chinese/English) worked examples explaining the code and underlying concepts:

| Tutorial | Covers |
|---|---|
| [`tutorials/tutorial.ipynb`](eda/tutorials/tutorial.ipynb) | Python, pandas, NetworkX, NumPy/SciPy (notebooks 01–03) |
| [`tutorials/tutorial_04.ipynb`](eda/tutorials/tutorial_04.ipynb) | `scipy.sparse`, graph heuristics, spectral graph theory, scikit-learn evaluation (notebook 04) |

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
│   ├── 01_overview.ipynb          # EDA notebooks (run in order)
│   ├── 02_structure.ipynb
│   ├── 03_compound_disease.ipynb
│   ├── 04_link_prediction.ipynb
│   ├── tutorials/
│   │   ├── tutorial.ipynb         # concepts behind notebooks 01–03
│   │   └── tutorial_04.ipynb      # concepts behind notebook 04
│   ├── utils.py                   # shared data-loading and graph utilities
│   ├── pixi.toml                  # environment spec
│   └── pixi.lock
└── README.md
```

---

## References

- Himmelstein, D. S. et al. (2017). *Systematic integration of biomedical knowledge
  prioritizes drugs for repurposing.* eLife.
  [doi:10.7554/eLife.26726](https://doi.org/10.7554/eLife.26726)
- Hetionet repository: <https://github.com/hetio/hetionet>
