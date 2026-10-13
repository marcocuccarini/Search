"""Compute and cache dense retriever scores (run once, on a GPU machine).

    python revision/compute_base_scores.py                       # all methods, all datasets, val+test
    python revision/compute_base_scores.py --methods BGE-large    # just the main retriever

Same encoding as ValidateExperiments.py / TestExperiments.py: "title text" as
document, model-specific prefixes from config.py, cosine similarity, batch 128.
Output: revision_results/scores/<ds>_<split>_<method>.json  ({qid: [score per node]})
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import DATASETS, REPO, load_items, load_paper_graphs  # noqa: E402
from scores import SCORES_DIR, score_path  # noqa: E402

sys.path.insert(0, REPO)
from config import MODEL_CONFIGS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+",
                    default=[m for m, c in MODEL_CONFIGS.items() if c["type"] == "dense"])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer, util
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(SCORES_DIR, exist_ok=True)

    for method in args.methods:
        cfg = MODEL_CONFIGS[method]
        model = None
        for ds in args.datasets:
            for split in args.splits:
                out = score_path(ds, split, method)
                if os.path.exists(out):
                    print(f"[skip] {out}")
                    continue
                if model is None:
                    print(f"--- Loading {method} ({cfg['model']}) on {device} ---")
                    model = SentenceTransformer(cfg["model"], device=device)
                items = load_items(ds, split)
                graphs = load_paper_graphs(ds, split)
                items = [it for it in items if it["id"] in graphs]
                docs, doc_id = [], {}
                for it in items:
                    for nd in graphs[it["id"]]["nodes"]:
                        t = nd["title"] + " " + nd["text"]
                        if t not in doc_id:
                            doc_id[t] = len(docs)
                            docs.append(t)
                q = model.encode([cfg.get("query_prefix", "") + it["question"] for it in items],
                                 convert_to_tensor=True, batch_size=args.batch_size,
                                 show_progress_bar=True)
                d = model.encode([cfg.get("doc_prefix", "") + x for x in docs],
                                 convert_to_tensor=True, batch_size=args.batch_size,
                                 show_progress_bar=True)
                res = {}
                for i, it in enumerate(items):
                    idx = [doc_id[nd["title"] + " " + nd["text"]] for nd in graphs[it["id"]]["nodes"]]
                    res[it["id"]] = util.cos_sim(q[i:i + 1], d[idx])[0].cpu().tolist()
                with open(out, "w") as f:
                    json.dump(res, f)
                print(f"[done] {out}")


if __name__ == "__main__":
    main()
