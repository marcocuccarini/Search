"""Ablation (entity / triple / both), query-unaware graph and random graph.

For every retriever, dataset, graph variant, method (BFS = 'bfs', NSP = 'ge')
and edge type, the configuration is tuned on the validation split with the
repo's own grid (config.BFS_CONFIGS / GE_CONFIGS) and selection metric
(config.PRIMARY_METRIC), then evaluated on the test split. Random graphs are
tuned on seed 0 and evaluated on --seeds seeds.

    python revision/run_experiments.py                        # all retrievers with cached scores
    python revision/run_experiments.py --methods BM25 BGE-large

Outputs (revision_results/):
  experiments/<ds>_<retriever>.csv   one row per variant x method x type
  significance/<ds>_<retriever>.csv  paired bootstrap tests (ablation, variant vs aware)
  perquery/           per-query metrics, used by kg_quality.py and bad_example.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (OUT, TYPES, ensure_hashseed, evaluate, mean,  # noqa: E402
                    paired_bootstrap, write_csv)
from core import DATASETS, REPO, load_items  # noqa: E402
from graph_variants import VARIANTS, get_graphs  # noqa: E402
from scores import available_methods, get_scores  # noqa: E402

sys.path.insert(0, REPO)
from config import BFS_CONFIGS, GE_CONFIGS, PRIMARY_METRIC  # noqa: E402

REPORT = ["Recall@3", "Recall@5", "FullRec@3", "FullRec@5", "NDCG@10", "MRR"]


def tune(items, graphs, scores, mode, gtype, metric):
    best, best_cfg = -1.0, None
    for cfg in (BFS_CONFIGS if mode == "bfs" else GE_CONFIGS):
        v = mean(evaluate(items, graphs, scores, mode, cfg, gtype), metric)
        if v > best:  # strict '>' = first best in grid order, as in the repo
            best, best_cfg = v, cfg
    return best_cfg, best


def main():
    ensure_hashseed()
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--metric", default=PRIMARY_METRIC, help="validation selection metric")
    ap.add_argument("--bootstrap", type=int, default=10000)
    a = ap.parse_args()

    perq_dir = os.path.join(OUT, "perquery")
    os.makedirs(perq_dir, exist_ok=True)
    for ds in a.datasets:
        val_items, test_items = load_items(ds, "val"), load_items(ds, "test")
        for method in a.methods or available_methods(ds, "test"):
            t0 = time.time()
            rows, sig = [], []
            aware_val = get_graphs("aware", ds, "val", val_items)
            aware_test = get_graphs("aware", ds, "test", test_items)
            vi = [it for it in val_items if it["id"] in aware_val]
            ti = [it for it in test_items if it["id"] in aware_test]
            s_val = get_scores(method, ds, "val", vi, aware_val)
            s_test = get_scores(method, ds, "test", ti, aware_test)
            qids = [it["id"] for it in ti]

            base = evaluate(ti, aware_test, s_test, "none", {}, "both")
            perq = {"qids": qids, "none": {m: [r[m] for r in base] for m in REPORT}}
            rows.append({"dataset": ds, "retriever": method, "variant": "none", "mode": "none",
                         "type": "-", "config": "", **{m: round(mean(base, m), 4) for m in REPORT}})

            for variant in a.variants:
                g_val = get_graphs(variant, ds, "val", vi, seed=0)
                g_tests = ([get_graphs(variant, ds, "test", ti, seed=s) for s in range(a.seeds)]
                           if variant == "random" else [get_graphs(variant, ds, "test", ti)])
                for mode in ("bfs", "ge"):
                    for gtype in TYPES:
                        cfg, vbest = tune(vi, g_val, s_val, mode, gtype, a.metric)
                        per_seed = [evaluate(ti, g, s_test, mode, cfg, gtype) for g in g_tests]
                        r = {"dataset": ds, "retriever": method, "variant": variant, "mode": mode,
                             "type": gtype, "config": json.dumps(cfg), f"val_{a.metric}": round(vbest, 4),
                             "seeds": len(per_seed)}
                        for m in REPORT:
                            vals = [mean(res, m) for res in per_seed]
                            r[m] = round(float(np.mean(vals)), 4)
                            if len(vals) > 1:
                                r[f"{m}_std"] = round(float(np.std(vals)), 4)
                        rows.append(r)
                        key = f"{variant}|{mode}|{gtype}"
                        perq[key] = {m: np.mean([[q[m] for q in res] for res in per_seed], 0).tolist()
                                     for m in REPORT}
                        perq[key]["config"] = cfg
                        print(f"{ds:9s} {method:9s} {variant:12s} {mode:3s} {gtype:6s} "
                              f"FR@5={r['FullRec@5']:.4f} R@5={r['Recall@5']:.4f}  cfg={cfg}", flush=True)

            # ---- significance: ablation within the aware graph, variants vs aware
            for mode in ("bfs", "ge"):
                comps = [(f"aware|{mode}|{x}", f"aware|{mode}|{y}", "ablation")
                         for x, y in (("triple", "both"), ("entity", "both"), ("triple", "entity"))]
                comps += [(f"{v}|{mode}|{t}", f"aware|{mode}|{t}", "variant_vs_aware")
                          for v in a.variants if v != "aware" for t in TYPES]
                comps += [(f"{v}|{mode}|both", "none", "vs_baseline") for v in a.variants]
                for x, y, kind in comps:
                    if x not in perq or y not in perq:
                        continue
                    for m in ("FullRec@5", "Recall@5"):
                        d, lo, hi, p = paired_bootstrap(perq[x][m], perq[y][m], B=a.bootstrap)
                        sig.append({"dataset": ds, "retriever": method, "kind": kind, "A": x, "B": y,
                                    "metric": m, "mean_A": round(float(np.mean(perq[x][m])), 4),
                                    "mean_B": round(float(np.mean(perq[y][m])), 4),
                                    "diff": round(d, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                                    "p_value": round(p, 4), "significant": int(p < 0.05)})
            with open(os.path.join(perq_dir, f"{ds}_{method}.json"), "w") as f:
                json.dump(perq, f)
            print(f"  [{ds} {method}] {time.time() - t0:.0f}s", flush=True)
            write_csv(os.path.join(OUT, "experiments", f"{ds}_{method}.csv"), rows)
            write_csv(os.path.join(OUT, "significance", f"{ds}_{method}.csv"), sig)


if __name__ == "__main__":
    main()
