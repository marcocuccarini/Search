"""The graph variants compared in the revision.

  aware         the paper's graphs (graphs/*_hybrid_graphs.json, query-aware KG)
  unaware       built with the repo's builder from dataKG_without (query-unaware KG)
  unaware_norm  same, with '_' in entity names replaced by spaces (the
                query-unaware KGs often use ids like 'Clara_Peeters', which the
                repo's substring matcher can never find in the passages)
  random        the aware graph with every connected passage pair moved to a
                random passage pair: same number of connected pairs, same edge
                multiplicities and edge types, random connections

Rebuilt graphs are cached in revision_results/graphs/.
"""
from __future__ import annotations

import json
import os
import zlib

import numpy as np

from core import REPO, build_graphs, load_kg, load_paper_graphs, normalise_underscores

CACHE = os.path.join(REPO, "revision_results", "graphs")
VARIANTS = ("aware", "unaware", "unaware_norm", "random")


def randomise_graph(graph, rng):
    """Move every connected (unordered) passage pair to a random pair."""
    n = len(graph["nodes"])
    by_pair = {}
    for e in graph.get("edges", []):
        key = (min(e["src"], e["dst"]), max(e["src"], e["dst"]))
        by_pair.setdefault(key, []).append(e)
    m = len(by_pair)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if m == 0 or n < 2:
        return {**graph, "edges": []}
    chosen = rng.choice(len(all_pairs), size=min(m, len(all_pairs)), replace=False)
    new_edges = []
    for (old, edges), c in zip(by_pair.items(), chosen):
        a, b = all_pairs[c]
        if rng.random() < 0.5:
            a, b = b, a
        for e in edges:
            ne = dict(e)
            # keep the edge's orientation relative to its old pair
            ne["src"], ne["dst"] = (a, b) if e["src"] == old[0] else (b, a)
            new_edges.append(ne)
    return {**graph, "edges": new_edges}


def get_graphs(variant, ds, split, items, seed=0):
    if variant == "aware":
        return load_paper_graphs(ds, split)
    if variant == "random":
        aware = load_paper_graphs(ds, split)
        out = {}
        for it in items:
            rng = np.random.default_rng([seed, zlib.crc32(it["id"].encode())])
            out[it["id"]] = randomise_graph(aware[it["id"]], rng)
        return out
    path = os.path.join(CACHE, f"{variant}_{ds}_{split}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    kg = load_kg("dataKG_without", ds, split)
    tr = normalise_underscores if variant == "unaware_norm" else None
    graphs = build_graphs(items, kg, tr)
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "w") as f:
        json.dump(graphs, f)
    return graphs
