"""Step 0: re-run the paper's best configurations on the test split and compare
with the appendix tables. Run this first: if it matches, everything else is
computed in exactly the paper's setting.

    python revision/reproduce.py                      # every retriever with scores available
    python revision/reproduce.py --methods BM25
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (OUT, TYPES, ensure_hashseed, evaluate, mean, paper_cfg,  # noqa: E402
                    paper_row, write_csv)
from core import DATASETS, load_items, load_paper_graphs  # noqa: E402
from scores import available_methods, get_scores  # noqa: E402

COLS = [("R@3", "Recall@3"), ("R@5", "Recall@5"), ("FR@3", "FullRec@3"), ("FR@5", "FullRec@5")]


def main():
    ensure_hashseed()
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    a = ap.parse_args()
    rows = []
    for ds in a.datasets:
        items = load_items(ds, "test")
        graphs = load_paper_graphs(ds, "test")
        items = [it for it in items if it["id"] in graphs]
        for method in a.methods or available_methods(ds, "test"):
            scores = get_scores(method, ds, "test", items, graphs)
            for mode in ("none", "bfs", "ge"):
                for gtype in (("-",) if mode == "none" else TYPES):
                    cfg = paper_cfg(ds, method, mode, gtype)
                    ref = paper_row(ds, method, mode, gtype)
                    if cfg is None or ref is None:
                        continue
                    res = evaluate(items, graphs, scores, mode, cfg, gtype if mode != "none" else "both")
                    r = {"dataset": ds, "method": method, "mode": mode, "type": gtype,
                         "config": str(cfg)}
                    for short, long in COLS:
                        r[f"paper_{short}"] = ref[short]
                        r[f"ours_{short}"] = round(mean(res, long), 4)
                    r["max_abs_diff"] = round(max(abs(r[f"paper_{s}"] - r[f"ours_{s}"]) for s, _ in COLS), 4)
                    rows.append(r)
                    print(f"{ds:9s} {method:9s} {mode:4s} {gtype:6s} FR@5 paper {ref['FR@5']:.4f} "
                          f"ours {r['ours_FR@5']:.4f}  max|diff| {r['max_abs_diff']:.4f}")
    write_csv(os.path.join(OUT, "reproduce.csv"), rows)
    bad = [r for r in rows if r["max_abs_diff"] > 0.002]
    print(f"\n{len(rows) - len(bad)}/{len(rows)} rows reproduce the paper within 0.002")
    for r in bad:
        print("  MISMATCH:", r["dataset"], r["method"], r["mode"], r["type"], r["max_abs_diff"])


if __name__ == "__main__":
    main()
