"""Base retriever scores, computed exactly as in ValidateExperiments.py / TestExperiments.py.

BM25 is computed on the fly (no model needed). Dense scores must be computed
once with compute_base_scores.py (GPU recommended) and are read from
revision_results/scores/<ds>_<split>_<method>.json.
"""
from __future__ import annotations

import json
import os

from core import REPO

SCORES_DIR = os.path.join(REPO, "revision_results", "scores")


def bm25_scores(items, graphs):
    from rank_bm25 import BM25Okapi
    out = {}
    for it in items:
        g = graphs[it["id"]]
        corpus = [nd["title"].lower().split() + nd["text"].lower().split() for nd in g["nodes"]]
        out[it["id"]] = BM25Okapi(corpus).get_scores(it["question"].lower().split()).tolist()
    return out


def score_path(ds, split, method):
    return os.path.join(SCORES_DIR, f"{ds}_{split}_{method}.json")


def get_scores(method, ds, split, items, graphs):
    path = score_path(ds, split, method)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    if method == "BM25":
        s = bm25_scores(items, graphs)
        os.makedirs(SCORES_DIR, exist_ok=True)
        with open(path, "w") as f:
            json.dump(s, f)
        return s
    raise SystemExit(f"No cached scores for {method} on {ds}/{split} ({path}).\n"
                     f"Run on a GPU machine:  python revision/compute_base_scores.py "
                     f"--methods {method}")


def available_methods(ds, split):
    meths = ["BM25"]
    if os.path.isdir(SCORES_DIR):
        for f in sorted(os.listdir(SCORES_DIR)):
            pre = f"{ds}_{split}_"
            if f.startswith(pre) and f.endswith(".json"):
                m = f[len(pre):-5]
                if m not in meths:
                    meths.append(m)
    return meths
