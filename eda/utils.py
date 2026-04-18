"""Shared utilities: Hetionet data loading, NetworkX conversion, subgraph construction."""
from __future__ import annotations

import bz2
import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

# Data path: override with HETIONET_PATH environment variable if needed.
_DEFAULT_DATA = (
    Path(__file__).parent.parent
    / "hetionet-main"
    / "hetnet"
    / "json"
    / "hetionet-v1.0.json.bz2"
)
DATA_PATH = Path(os.environ.get("HETIONET_PATH", _DEFAULT_DATA))


def setup_plot_style() -> None:
    """Apply seaborn whitegrid theme with a Unicode-safe font stack."""
    matplotlib.rcParams["font.family"] = [
        "Arial Unicode MS",
        "DejaVu Sans",
        "sans-serif",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 120


def load_hetnet(path: Path = DATA_PATH) -> dict:
    """Decompress and load the full Hetionet v1.0 JSON."""
    with bz2.open(path, "rt") as f:
        return json.load(f)


def node_key(node: dict) -> tuple[str, str | int]:
    """Globally unique node key: (kind, identifier)."""
    return (node["kind"], node["identifier"])


def to_networkx(hetnet: dict, directed: bool = True) -> nx.MultiDiGraph | nx.MultiGraph:
    """Convert the hetnet JSON to a NetworkX graph.

    Node IDs are (kind, identifier) tuples; edge attributes include kind and direction.
    """
    G: nx.MultiDiGraph | nx.MultiGraph = nx.MultiDiGraph() if directed else nx.MultiGraph()
    for n in hetnet["nodes"]:
        G.add_node(node_key(n), kind=n["kind"], name=n["name"], **n.get("data", {}))
    for e in hetnet["edges"]:
        src = tuple(e["source_id"])
        tgt = tuple(e["target_id"])
        G.add_edge(src, tgt, kind=e["kind"], direction=e["direction"], **e.get("data", {}))
    return G


def metanode_counts(hetnet: dict) -> pd.Series:
    """Return node counts per metanode kind, sorted descending."""
    c = Counter(n["kind"] for n in hetnet["nodes"])
    return pd.Series(c).sort_values(ascending=False)


def metaedge_counts(hetnet: dict) -> pd.Series:
    """Return edge counts per metaedge kind, sorted descending."""
    c = Counter(e["kind"] for e in hetnet["edges"])
    return pd.Series(c).sort_values(ascending=False)


def build_subgraph(G: nx.Graph, kinds: Iterable[str]) -> nx.Graph:
    """Return the induced subgraph containing only nodes of the specified kinds.

    Used to keep centrality / connectivity computations tractable by restricting
    the full 47k-node graph to the biologically relevant core.
    """
    kinds_set = set(kinds)
    nodes = [n for n, d in G.nodes(data=True) if d["kind"] in kinds_set]
    return G.subgraph(nodes).copy()


def degree_table(G: nx.Graph, top_n: int = 20) -> pd.DataFrame:
    """Return a DataFrame of the top-N nodes by degree with kind and name columns."""
    degs = dict(G.degree())
    rows = []
    for node, deg in sorted(degs.items(), key=lambda x: -x[1])[:top_n]:
        attrs = G.nodes[node]
        rows.append({
            "kind":       attrs.get("kind"),
            "name":       attrs.get("name"),
            "identifier": node[1],
            "degree":     deg,
        })
    return pd.DataFrame(rows)
