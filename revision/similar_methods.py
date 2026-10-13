"""Similar graph rerankers, run on exactly the same graphs, retriever scores and
validation/test protocol as Search+ (so the comparison isolates the ranking
algorithm):

  ppr   Personalized PageRank over the passage graph, personalised on the
        (min-max normalised) retriever scores. This is the ranking mechanism of
        HippoRAG / HippoRAG 2 and the PPR baseline of GraphER.
  gcs   Graph Cohesive Smoothing from GraphER (Miao et al., 2026, arXiv:2603.24925):
        s <- (1-a)*s0 + a*P s iterated to convergence (P = row-normalised weighted
        adjacency), final score = max(s, s0). Re-implemented from the paper's
        description (no code was released), scores min-max normalised per query.

Both are tuned on the validation split (repo selection metric), evaluated on test.

    python revision/similar_methods.py --methods BM25 BGE-large
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (OUT, TYPES, ensure_hashseed, mean, paired_bootstrap,  # noqa: E402
                    write_csv)
from core import (DATASETS, REPO, eval_ranking, filter_edges_by_type, load_items,  # noqa: E402
                  supporting)
from graph_variants import get_graphs  # noqa: E402
from scores import available_methods, get_scores  # noqa: E402

sys.path.insert(0, REPO)
from config import PRIMARY_METRIC  # noqa: E402

GRID = {"ppr": [{"damping": d} for d in (0.3, 0.5, 0.7, 0.85, 0.95, 0.99)],
        "gcs": [{"a": a} for a in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)]}
REPORT = ["Recall@3", "Recall@5", "FullRec@3", "FullRec@5", "NDCG@10", "MRR"]


def weight_matrix(graph, gtype):
    n = len(graph["nodes"])
    W = np.zeros((n, n))
    for e in filter_edges_by_type(graph.get("edges", []), gtype):
        W[e["src"], e["dst"]] += 1.0  # same weighting as the Search+ code
        W[e["dst"], e["src"]] += 1.0
    return W


def minmax(s):
    s = np.asarray(s, float)
    lo, hi = s.min(), s.max()
    return np.full_like(s, 1.0) if hi - lo < 1e-12 else (s - lo) / (hi - lo)


def row_norm(W):
    deg = W.sum(1, keepdims=True)
    return np.divide(W, deg, out=np.zeros_like(W), where=deg > 0), deg[:, 0] > 0


def ppr_scores(W, base, damping, iters=100, tol=1e-10):
    n = len(base)
    p = minmax(base) + 1e-6
    p = p / p.sum()
    P, has = row_norm(W)
    r = p.copy()
    for _ in range(iters):
        spread = r[has] @ P[has]                    # mass leaving nodes with edges
        dangling = r[~has].sum()                    # mass of isolated nodes -> teleport
        new = (1 - damping) * p + damping * (spread + dangling * p)
        if np.abs(new - r).sum() < tol:
            r = new
            break
        r = new
    return r


def gcs_scores(W, base, a, iters=200, tol=1e-10):
    s0 = minmax(base)
    P, has = row_norm(W)
    s = s0.copy()
    for _ in range(iters):
        new = np.where(has, (1 - a) * s0 + a * (P @ s), s0)
        if np.abs(new - s).max() < tol:
            s = new
            break
        s = new
    return np.maximum(s, s0)


def run(items, graphs, scores, algo, cfg, gtype):
    out = []
    for it in items:
        g, base = graphs[it["id"]], scores[it["id"]]
        W = weight_matrix(g, gtype)
        sc = ppr_scores(W, base, **cfg) if algo == "ppr" else gcs_scores(W, base, **cfg)
        # ties broken by the retriever order (stable sort on the base ranking)
        order0 = sorted(range(len(base)), key=lambda i: base[i], reverse=True)
        ranked = sorted(order0, key=lambda i: -sc[i])
        out.append(eval_ranking(ranked, supporting(g), len(base)))
    return out


def main():
    ensure_hashseed()
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--metric", default=PRIMARY_METRIC)
    a = ap.parse_args()
    for ds in a.datasets:
        vi_all, ti_all = load_items(ds, "val"), load_items(ds, "test")
        gv, gt = get_graphs("aware", ds, "val", vi_all), get_graphs("aware", ds, "test", ti_all)
        vi = [it for it in vi_all if it["id"] in gv]
        ti = [it for it in ti_all if it["id"] in gt]
        for method in a.methods or available_methods(ds, "test"):
            rows, sig = [], []
            sv, st = get_scores(method, ds, "val", vi, gv), get_scores(method, ds, "test", ti, gt)
            perq_path = os.path.join(OUT, "perquery", f"{ds}_{method}.json")
            perq = json.load(open(perq_path)) if os.path.exists(perq_path) else None
            for algo in ("ppr", "gcs"):
                for gtype in TYPES:
                    best, best_cfg = -1, None
                    for cfg in GRID[algo]:
                        v = mean(run(vi, gv, sv, algo, cfg, gtype), a.metric)
                        if v > best:
                            best, best_cfg = v, cfg
                    res = run(ti, gt, st, algo, best_cfg, gtype)
                    r = {"dataset": ds, "retriever": method, "method": algo, "type": gtype,
                         "config": json.dumps(best_cfg), f"val_{a.metric}": round(best, 4),
                         **{m: round(mean(res, m), 4) for m in REPORT}}
                    rows.append(r)
                    print(f"{ds:9s} {method:9s} {algo} {gtype:6s} FR@5={r['FullRec@5']:.4f} "
                          f"R@5={r['Recall@5']:.4f} cfg={best_cfg}", flush=True)
                    if perq is not None:  # compare with Search+ BFS / NSP on the same queries
                        for ours in ("bfs", "ge"):
                            key = f"aware|{ours}|{gtype}"
                            if key not in perq:
                                continue
                            for m in ("FullRec@5", "Recall@5"):
                                d, lo, hi, p = paired_bootstrap(perq[key][m], [q[m] for q in res])
                                sig.append({"dataset": ds, "retriever": method, "A": key,
                                            "B": f"{algo}|{gtype}", "metric": m, "diff": round(d, 4),
                                            "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                                            "p_value": round(p, 4), "significant": int(p < 0.05)})
            write_csv(os.path.join(OUT, "similar_methods", f"{ds}_{method}.csv"), rows)
            write_csv(os.path.join(OUT, "similar_methods", f"significance_{ds}_{method}.csv"), sig)


if __name__ == "__main__":
    main()
