"""Core Search+ functions used by the revision experiments.

The reranking and metric functions below are VERBATIM copies of the ones in
TestExperiments.py / ValidateExperiments.py (same code, only type hints
dropped), so that the experiments run without torch / sentence-transformers.
Graphs are built with BuildGraphsPerQuestion.build_hybrid_pseudo_graph
(imported, not copied).
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from BuildGraphsPerQuestion import build_hybrid_pseudo_graph  # noqa: E402

DATASETS = ("2wiki", "hotpotqa", "musique")


# ═════════════════════════════════ metrics (verbatim) ═════════════════════════
def recall_at_k(ranked, k):
    n_rel = sum(ranked)
    if n_rel == 0:
        return 0.0
    return min(sum(ranked[:k]), n_rel) / n_rel


def full_recall_at_k(ranked, k):
    n_rel = sum(ranked)
    if n_rel == 0:
        return 0.0
    return 1.0 if sum(ranked[:k]) >= n_rel else 0.0


def ndcg_at_k(ranked, k):
    def dcg(r, kk):
        return sum(rel / math.log2(i + 2) for i, rel in enumerate(r[:kk]))
    ideal = sorted(ranked, reverse=True)
    d = dcg(ranked, k)
    d_ideal = dcg(ideal, k)
    return d / d_ideal if d_ideal > 0 else 0.0


def mrr(ranked):
    for i, r in enumerate(ranked):
        if r:
            return 1.0 / (i + 1)
    return 0.0


def eval_ranking(ranked, supp, n):
    ranked_bool = [i in supp for i in ranked]
    seen = set(ranked)
    for i in range(n):
        if i not in seen:
            ranked_bool.append(i in supp)
    return {
        "Recall@3": recall_at_k(ranked_bool, 3),
        "Recall@5": recall_at_k(ranked_bool, 5),
        "Recall@10": recall_at_k(ranked_bool, 10),
        "FullRec@3": full_recall_at_k(ranked_bool, 3),
        "FullRec@5": full_recall_at_k(ranked_bool, 5),
        "FullRec@10": full_recall_at_k(ranked_bool, 10),
        "NDCG@10": ndcg_at_k(ranked_bool, 10),
        "MRR": mrr(ranked_bool),
    }


METRICS = ["Recall@3", "Recall@5", "Recall@10", "FullRec@3", "FullRec@5",
           "FullRec@10", "NDCG@10", "MRR"]


def avg_metrics(results):
    return {k: round(sum(r[k] for r in results) / len(results), 4) if results else 0.0
            for k in METRICS}


# ═════════════════════════════ graph algorithms (verbatim) ════════════════════
def filter_edges_by_type(edges, mode):
    if mode == "triple":
        return [e for e in edges if e.get("relation_type") == "kg_bridge"]
    elif mode == "entity":
        return [e for e in edges if e.get("relation_type") == "entity_overlap"]
    return edges  # "both" / "all"


def compute_ge_embeddings(graph, base_scores, ge_alpha, ge_hops, graph_type):
    nodes = graph["nodes"]
    edges = filter_edges_by_type(graph.get("edges", []), graph_type)
    n = len(nodes)
    adj_strength = defaultdict(lambda: defaultdict(float))
    for e in edges:
        src, dst = e["src"], e["dst"]
        adj_strength[src][dst] += 1.0
        adj_strength[dst][src] += 1.0
    adj = defaultdict(list)
    for src in adj_strength:
        for dst, strength in adj_strength[src].items():
            adj[src].append((dst, strength))
    scores = list(base_scores)
    for _ in range(ge_hops):
        new_scores = []
        for i in range(n):
            neighs = adj[i]
            if not neighs or ge_alpha == 0.0:
                new_scores.append(scores[i])
                continue
            total_w = sum(w for _, w in neighs)
            neigh_avg = sum(scores[j] * w for j, w in neighs) / total_w
            new_scores.append((1 - ge_alpha) * scores[i] + ge_alpha * neigh_avg)
        scores = new_scores
    return scores


def bfs_retrieve_with_base(graph, base_scores, top_k_bm25, max_depth, branch, alpha, graph_type):
    nodes = graph["nodes"]
    edges = filter_edges_by_type(graph.get("edges", []), graph_type)
    n = len(nodes)
    adj_strength = defaultdict(lambda: defaultdict(float))
    for e in edges:
        src, dst = e["src"], e["dst"]
        adj_strength[src][dst] += 1.0
        adj_strength[dst][src] += 1.0
    adj = defaultdict(list)
    for src in adj_strength:
        for dst, strength in adj_strength[src].items():
            adj[src].append((dst, strength))
    ranked_base = sorted(range(n), key=lambda i: base_scores[i], reverse=True)
    seeds = ranked_base[:top_k_bm25]
    visited, v_set, frontier = list(seeds), set(seeds), list(seeds)
    max_s = max(base_scores) if max(base_scores) > 0 else 1.0
    for _ in range(max_depth):
        if not frontier:
            break
        next_f = []
        for node in frontier:
            scored = []
            for j, w in adj[node]:
                if j in v_set:
                    continue
                score = alpha * (w / n) + (1 - alpha) * (base_scores[j] / max_s)
                scored.append((j, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            for j, _ in scored[:branch]:
                if j not in v_set:
                    v_set.add(j)
                    visited.append(j)
                    next_f.append(j)
        frontier = next_f
    for i in ranked_base:
        if i not in v_set:
            visited.append(i)
    return visited


# ═════════════════════════════════ ranking dispatch ═══════════════════════════
def rank(mode, graph, scores, cfg, graph_type):
    """mode: none | bfs | ge  (exactly as in TestExperiments.py)."""
    if mode == "none":
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    if mode == "bfs":
        return bfs_retrieve_with_base(graph, scores, **cfg, graph_type=graph_type)
    ge = compute_ge_embeddings(graph, scores, **cfg, graph_type=graph_type)
    return sorted(range(len(ge)), key=lambda i: ge[i], reverse=True)


def supporting(graph):
    return {i for i, nd in enumerate(graph["nodes"]) if nd["is_supporting"]}


# ═══════════════════════════════════════ I/O ══════════════════════════════════
def load_items(ds, split):
    path = os.path.join(REPO, "data", f"{ds}_validation_{split}.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_kg(kg_dir, ds, split):
    with open(os.path.join(REPO, kg_dir, f"{ds}_validation_{split}_KG.json"), encoding="utf-8") as f:
        return json.load(f)


def load_paper_graphs(ds, split):
    """The hybrid graphs used for the paper (query-aware KG)."""
    with open(os.path.join(REPO, "graphs", f"{ds}_validation_{split}_hybrid_graphs.json"),
              encoding="utf-8") as f:
        return json.load(f)


def build_graphs(items, kg_map, transform=None):
    """Build hybrid graphs with the repo's own builder; optional KG transform."""
    out = {}
    for it in items:
        kg = kg_map.get(it["id"])
        if transform is not None and kg is not None:
            kg = transform(kg)
        out[it["id"]] = build_hybrid_pseudo_graph(it["id"], it, kg)
    return out


def normalise_underscores(kg_item):
    """'Clara_Peeters' -> 'Clara Peeters' in triple endpoints, so that the
    repo's substring matcher can find them in the passages."""
    kg = dict(kg_item.get("knowledge_graph", kg_item))
    edges = []
    for e in kg.get("edges", []) or []:
        if isinstance(e, dict):
            e = {k: (v.replace("_", " ") if isinstance(v, str) and k in
                     ("source", "target", "entity", "entity2") else v) for k, v in e.items()}
        elif isinstance(e, list):
            e = [x.replace("_", " ") if isinstance(x, str) else x for x in e]
        edges.append(e)
    kg["edges"] = edges
    return {"knowledge_graph": kg}
