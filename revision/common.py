"""Shared helpers for the revision experiments."""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

from core import REPO, eval_ranking, rank, supporting
from paper_tables import find_row, parse_tables

OUT = os.path.join(REPO, "revision_results")
PAPER_TEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper.tex")
TYPES = ("entity", "triple", "both")
MODE_TABLE = {"bfs": "BFS", "ge": "NSP", "none": "Baseline"}


def ensure_hashseed():
    """Re-exec with PYTHONHASHSEED=0 so graphs rebuilt with the repo's builder
    (which iterates over a set) have a reproducible edge order."""
    if os.environ.get("PYTHONHASHSEED") != "0":
        import subprocess
        env = dict(os.environ, PYTHONHASHSEED="0")
        # subprocess instead of os.execv: os.execv does not replace the process on Windows
        sys.exit(subprocess.call([sys.executable] + sys.argv, env=env))


def paper_cfg(ds, method, mode, gtype):
    """Best config from the paper's appendix table, in the repo's parameter names."""
    if mode == "none":
        return {}
    row = find_row(parse_tables(PAPER_TEX)[ds], method, MODE_TABLE[mode], gtype)
    if row is None:
        return None
    p = row["params"]
    if mode == "bfs":
        return {"top_k_bm25": int(p["h"]), "max_depth": int(p["d"]),
                "branch": int(p["b"]), "alpha": float(p["alpha"])}
    return {"ge_alpha": float(p["alpha"]), "ge_hops": int(p["hops"])}


def paper_row(ds, method, mode, gtype):
    return find_row(parse_tables(PAPER_TEX)[ds], method, MODE_TABLE[mode], gtype)


def evaluate(items, graphs, scores, mode, cfg, gtype):
    """Per-query metric dicts, in item order (exactly the repo's evaluation)."""
    out = []
    for it in items:
        g = graphs[it["id"]]
        s = scores[it["id"]]
        out.append(eval_ranking(rank(mode, g, s, cfg, gtype), supporting(g), len(s)))
    return out


def mean(rows, k):
    return float(np.mean([r[k] for r in rows])) if rows else float("nan")


def paired_bootstrap(a, b, B=10000, seed=0):
    """mean(a-b), 95% CI, two-sided p-value, paired over queries."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    n = len(d)
    boots = np.concatenate([d[rng.integers(0, n, (min(1000, B - s), n))].mean(1)
                            for s in range(0, B, 1000)])
    p = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), float(p)


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {os.path.relpath(path, REPO)}")


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    print(f"  -> {os.path.relpath(path, REPO)}")
