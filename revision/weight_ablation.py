"""Edge-weight ablation (reviewer Ttyg: "a discussion on this choice must be made").

Same protocol as run_experiments.py / similar_methods.py (same graphs, retriever
scores, grids, selection metric, strict-'>' tuning on validation, test
evaluation), with one extra axis: how each edge record is weighted.

The repo's rankers add +1 per edge record in both directions, i.e.
    w_ij = B_ij + 2|S_ij|                                   (scheme "count")
This script compares it with:
    binary   w_ij = 1 if the pair is connected                (no weight)
    lam0.5   w_ij = B_ij + 0.5|S_ij|   } entity factor lambda instead of 2
    lam1     w_ij = B_ij + 1  |S_ij|   } (only on 'both' graphs: on entity-only
    lam4     w_ij = B_ij + 4  |S_ij|   }  graphs lambda just rescales)
    idf      w_ij = B_ij + 2 sum_{e in S_ij} log(n / n_e)
             n_e = passages of the pool containing e: generic entities shared
             by many passages count less (not on triple-only graphs)

Everything else is the repo's code: BFS and NSP below are core.py's
bfs_retrieve_with_base / compute_ge_embeddings with only the "+= 1.0" line
replaced by the scheme's weight (same adjacency order, hence same
tie-breaking); PPR and GCS are similar_methods.ppr_scores / gcs_scores. With
--check, the "count" rows are compared with experiments/ and
similar_methods/ and must match.

BFS diagnostics (how often the weight matters at all):
    share_over_b          expansions with more than b candidate neighbours
    share_weight_decides  expansions whose kept b neighbours differ from those
                          kept with binary weights (same config)

    python revision/weight_ablation.py --methods BM25 BGE-large
    python revision/weight_ablation.py --variants aware unaware_norm --rankers bfs ge
    python revision/weight_ablation.py --methods BM25 --datasets 2wiki --check

Outputs (revision_results/weight_ablation/):
    <ds>_<retriever>.csv               one row per variant x ranker x type x scheme
    significance_<ds>_<retriever>.csv  paired bootstrap, every scheme vs count
    perquery_<ds>_<retriever>.json
    summary_<metric>.csv               mean over retrievers (rows: variant/type/
                                       scheme, columns: dataset|ranker)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, TYPES, ensure_hashseed, mean, paired_bootstrap, write_csv  # noqa: E402
from core import (DATASETS, eval_ranking, filter_edges_by_type, load_items,  # noqa: E402
                  supporting)
from graph_variants import get_graphs  # noqa: E402
from scores import available_methods, get_scores  # noqa: E402
from similar_methods import GRID as SIM_GRID, gcs_scores, ppr_scores  # noqa: E402

from core import REPO  # noqa: E402
sys.path.insert(0, REPO)
from config import BFS_CONFIGS, GE_CONFIGS, PRIMARY_METRIC  # noqa: E402

SCHEMES = ("count", "binary", "lam0.5", "lam1", "lam4", "idf")
RANKERS = ("bfs", "ge", "ppr", "gcs")
REPORT = ["Recall@3", "Recall@5", "FullRec@3", "FullRec@5", "NDCG@10", "MRR"]
WA_OUT = os.path.join(OUT, "weight_ablation")


def applies(scheme, gtype):
    """Skip combinations that are identical to 'count' by construction."""
    if scheme.startswith("lam"):
        return gtype == "both"
    if scheme == "idf":
        return gtype != "triple"
    return True


# ═══════════════════════════════ weighted adjacency ═══════════════════════════
def entity_df(edges):
    """n_e: number of passages that share entity e with some other passage."""
    nodes = defaultdict(set)
    for e in edges:
        if e.get("relation_type") == "entity_overlap":
            nodes[e["shared_entity"]].update((e["src"], e["dst"]))
    return {k: len(v) for k, v in nodes.items()}


def record_weight(e, scheme, df, n):
    """Contribution of ONE edge record (added in both directions, as in core.py).
    Entity-overlap records come in pairs (i->j, j->i), so a shared entity
    contributes 2x this value to w_ij."""
    if e.get("relation_type") == "entity_overlap":
        if scheme.startswith("lam"):
            return float(scheme[3:]) / 2.0
        if scheme == "idf":
            return max(math.log(n / max(df.get(e["shared_entity"], 2), 1)), 1e-6)
    return 1.0


def weighted_adj(graph, gtype, scheme):
    """core.py's adjacency construction with the weight made explicit.
    Returns adj[node] = [(neighbour, w), ...] in the repo's insertion order."""
    n = len(graph["nodes"])
    edges = filter_edges_by_type(graph.get("edges", []), gtype)
    df = entity_df(edges) if scheme == "idf" else None
    adj_strength = defaultdict(lambda: defaultdict(float))
    for e in edges:
        src, dst = e["src"], e["dst"]
        w = record_weight(e, scheme, df, n)
        adj_strength[src][dst] += w
        adj_strength[dst][src] += w
    adj = defaultdict(list)
    for src in adj_strength:
        for dst, strength in adj_strength[src].items():
            adj[src].append((dst, 1.0 if scheme == "binary" else strength))
    return adj


def adj_to_matrix(adj, n):
    W = np.zeros((n, n))
    for i, lst in adj.items():
        for j, w in lst:
            W[i, j] = w
    return W


# ═════════════════════ rankers (repo code, adjacency passed in) ═══════════════
def bfs_rank(adj, n, base_scores, top_k_bm25, max_depth, branch, alpha,
             bfs_norm="n", stats=None):
    """core.bfs_retrieve_with_base; bfs_norm='max' divides w by max_k w_uk."""
    ranked_base = sorted(range(n), key=lambda i: base_scores[i], reverse=True)
    seeds = ranked_base[:top_k_bm25]
    visited, v_set, frontier = list(seeds), set(seeds), list(seeds)
    max_s = max(base_scores) if max(base_scores) > 0 else 1.0
    for _ in range(max_depth):
        if not frontier:
            break
        next_f = []
        for node in frontier:
            den = n if bfs_norm == "n" else (max((w for _, w in adj[node]), default=1.0) or 1.0)
            scored = []
            for j, w in adj[node]:
                if j in v_set:
                    continue
                score = alpha * (w / den) + (1 - alpha) * (base_scores[j] / max_s)
                scored.append((j, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            if stats is not None and scored:
                stats["expansions"] += 1
                if len(scored) > branch:
                    stats["over_b"] += 1
                    if alpha > 0:
                        # same candidates (adjacency order), ranked as if every w were 1
                        cand = [j for j, _ in adj[node] if j not in v_set]
                        binary = (cand if alpha == 1 else
                                  sorted(cand, key=lambda j: base_scores[j], reverse=True))
                        if set(binary[:branch]) != {j for j, _ in scored[:branch]}:
                            stats["weight_decides"] += 1
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


def ge_rank(adj, n, base_scores, ge_alpha, ge_hops):
    """core.compute_ge_embeddings + core.rank('ge')."""
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
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


def sim_rank(algo, W, base, cfg):
    """similar_methods.run, single query."""
    sc = ppr_scores(W, base, **cfg) if algo == "ppr" else gcs_scores(W, base, **cfg)
    order0 = sorted(range(len(base)), key=lambda i: base[i], reverse=True)
    return sorted(order0, key=lambda i: -sc[i])


def grid(ranker):
    return {"bfs": BFS_CONFIGS, "ge": GE_CONFIGS}.get(ranker) or SIM_GRID[ranker]


def evaluate(items, graphs, scores, prepared, ranker, cfg, bfs_norm, stats=None):
    out = []
    for it in items:
        q = it["id"]
        g, s = graphs[q], scores[q]
        n = len(s)
        if ranker == "bfs":
            r = bfs_rank(prepared[q], n, s, **cfg, bfs_norm=bfs_norm, stats=stats)
        elif ranker == "ge":
            r = ge_rank(prepared[q], n, s, **cfg)
        else:
            r = sim_rank(ranker, prepared[q], s, cfg)
        out.append(eval_ranking(r, supporting(g), n))
    return out


def tune(items, graphs, scores, prepared, ranker, metric, bfs_norm):
    best, best_cfg = -1.0, None
    for cfg in grid(ranker):
        v = mean(evaluate(items, graphs, scores, prepared, ranker, cfg, bfs_norm), metric)
        if v > best:  # strict '>' = first best in grid order, as in the repo
            best, best_cfg = v, cfg
    return best_cfg, best


def prepare(items, graphs, gtype, scheme, ranker):
    out = {}
    for it in items:
        g = graphs[it["id"]]
        adj = weighted_adj(g, gtype, scheme)
        out[it["id"]] = adj if ranker in ("bfs", "ge") else adj_to_matrix(adj, len(g["nodes"]))
    return out


# ═════════════════════════════════════ check ══════════════════════════════════
def check_against_repo(rows):
    """'count' rows must equal experiments/ (bfs, ge) and similar_methods/ (ppr, gcs)."""
    bad, n = [], 0
    for r in rows:
        if r["scheme"] != "count":
            continue
        if r["ranker"] in ("bfs", "ge"):
            p = os.path.join(OUT, "experiments", f"{r['dataset']}_{r['retriever']}.csv")
            match = lambda x: (x["variant"] == r["variant"] and x["mode"] == r["ranker"]  # noqa: E731
                               and x["type"] == r["type"])
        else:
            if r["variant"] != "aware":
                continue
            p = os.path.join(OUT, "similar_methods", f"{r['dataset']}_{r['retriever']}.csv")
            match = lambda x: x["method"] == r["ranker"] and x["type"] == r["type"]  # noqa: E731
        if not os.path.exists(p):
            continue
        ref = next((x for x in csv.DictReader(open(p)) if match(x)), None)
        if ref is None:
            continue
        n += 1
        d = max(abs(float(ref[m]) - r[m]) for m in ("Recall@5", "FullRec@5"))
        if d > 0.002:
            bad.append((r["dataset"], r["retriever"], r["variant"], r["ranker"], r["type"], round(d, 4)))
    print(f"\n[check] {n - len(bad)}/{n} 'count' rows match the existing results (|diff| <= 0.002)")
    for b in bad:
        print("  MISMATCH", *b)


# ═════════════════════════════════════ summary ════════════════════════════════
def write_summaries(suffix="", metrics=("FullRec@5", "Recall@5")):
    rows = []
    for f in sorted(os.listdir(WA_OUT)):
        if (f.endswith(".csv") and not f.startswith(("significance_", "summary_"))
                and f.endswith("_bfsmax.csv") == (suffix == "_bfsmax")):
            rows += list(csv.DictReader(open(os.path.join(WA_OUT, f))))
    for m in metrics:
        agg = defaultdict(list)
        for r in rows:
            agg[(r["variant"], r["type"], r["scheme"], r["dataset"], r["ranker"])].append(float(r[m]))
        keys = sorted({k[:3] for k in agg}, key=lambda k: (k[0], TYPES.index(k[1]), SCHEMES.index(k[2])))
        cols = sorted({k[3:] for k in agg}, key=lambda c: (DATASETS.index(c[0]), RANKERS.index(c[1])))
        out = []
        for k in keys:
            row = {"variant": k[0], "type": k[1], "scheme": k[2]}
            for c in cols:
                v = agg.get(k + c)
                row[f"{c[0]}|{c[1]}"] = round(100 * float(np.mean(v)), 1) if v else ""
                row[f"{c[0]}|{c[1]}|n"] = len(v) if v else 0
            out.append(row)
        write_csv(os.path.join(WA_OUT, f"summary_{m}{suffix}.csv"), out)


# ═══════════════════════════════════════ main ═════════════════════════════════
def main():
    ensure_hashseed()
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--variants", nargs="+", default=["aware"],
                    help="aware unaware unaware_norm (random is not meaningful here)")
    ap.add_argument("--rankers", nargs="+", default=list(RANKERS), choices=RANKERS)
    ap.add_argument("--schemes", nargs="+", default=list(SCHEMES), choices=SCHEMES)
    ap.add_argument("--bfs-norm", choices=("n", "max"), default="n",
                    help="'n' = paper (w/n); 'max' = w / max_k w_uk")
    ap.add_argument("--metric", default=PRIMARY_METRIC, help="validation selection metric")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--check", action="store_true",
                    help="compare the 'count' rows with experiments/ and similar_methods/")
    a = ap.parse_args()
    os.makedirs(WA_OUT, exist_ok=True)
    suffix = "" if a.bfs_norm == "n" else "_bfsmax"
    all_rows = []

    for ds in a.datasets:
        val_all, test_all = load_items(ds, "val"), load_items(ds, "test")
        graphs = {}
        for v in a.variants:
            gv = get_graphs(v, ds, "val", val_all)
            gt = get_graphs(v, ds, "test", test_all)
            graphs[v] = (gv, gt)
        aware_v, aware_t = get_graphs("aware", ds, "val", val_all), get_graphs("aware", ds, "test", test_all)
        vi = [it for it in val_all if it["id"] in aware_v]   # same query sets as run_experiments.py
        ti = [it for it in test_all if it["id"] in aware_t]

        for method in a.methods or available_methods(ds, "test"):
            t0 = time.time()
            s_val = get_scores(method, ds, "val", vi, aware_v)
            s_test = get_scores(method, ds, "test", ti, aware_t)
            rows, sig = [], []
            perq = {"qids": [it["id"] for it in ti]}
            for v in a.variants:
                gv, gt = graphs[v]
                for gtype in TYPES:
                    for scheme in a.schemes:
                        if not applies(scheme, gtype):
                            continue
                        for ranker in a.rankers:
                            pv = prepare(vi, gv, gtype, scheme, ranker)
                            pt = prepare(ti, gt, gtype, scheme, ranker)
                            cfg, vbest = tune(vi, gv, s_val, pv, ranker, a.metric, a.bfs_norm)
                            stats = defaultdict(int) if ranker == "bfs" else None
                            res = evaluate(ti, gt, s_test, pt, ranker, cfg, a.bfs_norm, stats)
                            r = {"dataset": ds, "retriever": method, "variant": v, "ranker": ranker,
                                 "type": gtype, "scheme": scheme, "config": json.dumps(cfg),
                                 f"val_{a.metric}": round(vbest, 4),
                                 **{m: round(mean(res, m), 4) for m in REPORT}}
                            if stats is not None and stats["expansions"]:
                                r["share_over_b"] = round(stats["over_b"] / stats["expansions"], 4)
                                r["share_weight_decides"] = ("" if scheme == "binary" else
                                                             round(stats["weight_decides"] / stats["expansions"], 4))
                            rows.append(r)
                            perq[f"{v}|{ranker}|{gtype}|{scheme}"] = {m: [q[m] for q in res] for m in REPORT}
                            print(f"{ds:9s} {method:9s} {v:12s} {ranker:3s} {gtype:6s} {scheme:6s} "
                                  f"FR@5={r['FullRec@5']:.4f} R@5={r['Recall@5']:.4f} cfg={cfg}", flush=True)

            # every scheme vs count, same variant / ranker / edge type
            for key, val in perq.items():
                if key == "qids":
                    continue
                v, ranker, gtype, scheme = key.split("|")
                ref = perq.get(f"{v}|{ranker}|{gtype}|count")
                if scheme == "count" or ref is None:
                    continue
                for m in ("FullRec@5", "Recall@5"):
                    d, lo, hi, p = paired_bootstrap(val[m], ref[m], B=a.bootstrap)
                    sig.append({"dataset": ds, "retriever": method, "variant": v, "ranker": ranker,
                                "type": gtype, "A": scheme, "B": "count", "metric": m,
                                "mean_A": round(float(np.mean(val[m])), 4),
                                "mean_B": round(float(np.mean(ref[m])), 4),
                                "diff": round(d, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                                "p_value": round(p, 4), "significant": int(p < 0.05)})

            write_csv(os.path.join(WA_OUT, f"{ds}_{method}{suffix}.csv"), rows)
            write_csv(os.path.join(WA_OUT, f"significance_{ds}_{method}{suffix}.csv"), sig)
            with open(os.path.join(WA_OUT, f"perquery_{ds}_{method}{suffix}.json"), "w") as f:
                json.dump(perq, f)
            all_rows += rows
            print(f"  [{ds} {method}] {time.time() - t0:.0f}s", flush=True)

    if a.check and a.bfs_norm == "n":
        check_against_repo(all_rows)
    write_summaries(suffix)


if __name__ == "__main__":
    main()
