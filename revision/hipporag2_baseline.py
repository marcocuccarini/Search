"""HippoRAG 2 in the Search+ setting: each question's own candidate pool is the
corpus, HippoRAG 2 ranks all its passages, evaluated with the repo's metrics.

Install in a SEPARATE virtualenv (HippoRAG pins its own torch/vllm):
    pip install hipporag
    ollama serve                          # OpenAI-compatible API at :11434/v1

    python revision/hipporag2_baseline.py --dataset musique --limit 20   # quick test
    python revision/hipporag2_baseline.py --dataset musique

CHECK against the README of the HippoRAG version you install: this uses the API
HippoRAG(save_dir, llm_model_name, llm_base_url, embedding_model_name),
.index(docs), .retrieve(queries, num_to_retrieve) -> objects with .docs.
If the constructor rejects an argument the script prints the real signature.
Use the same LLM as Search+ (Gemma 4) and report the embedding model you use.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, mean, write_csv  # noqa: E402
from core import eval_ranking, load_items, load_paper_graphs, supporting  # noqa: E402

REPORT = ["Recall@3", "Recall@5", "FullRec@3", "FullRec@5", "NDCG@10", "MRR"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["2wiki", "hotpotqa", "musique"])
    ap.add_argument("--llm", default="gemma4:latest")
    ap.add_argument("--llm-base-url", default="http://localhost:11434/v1")
    ap.add_argument("--embedding", default="nvidia/NV-Embed-v2")
    ap.add_argument("--workdir", default="hipporag_tmp")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.environ.setdefault("OPENAI_API_KEY", "ollama")
    from hipporag import HippoRAG

    graphs = load_paper_graphs(a.dataset, "test")
    items = [it for it in load_items(a.dataset, "test") if it["id"] in graphs]
    if a.limit:
        items = items[:a.limit]
    out_dir = os.path.join(OUT, "hipporag2")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{a.dataset}.jsonl")
    done = {json.loads(l)["id"] for l in open(path)} if os.path.exists(path) else set()

    with open(path, "a") as f:
        for k, it in enumerate(items):
            if it["id"] in done:
                continue
            g = graphs[it["id"]]
            docs = [nd["title"] + "\n" + nd["text"] for nd in g["nodes"]]
            save = os.path.join(a.workdir, it["id"].replace("/", "_"))
            shutil.rmtree(save, ignore_errors=True)
            t0 = time.perf_counter()
            kwargs = dict(save_dir=save, llm_model_name=a.llm, llm_base_url=a.llm_base_url,
                          embedding_model_name=a.embedding)
            try:
                hr = HippoRAG(**kwargs)
            except TypeError:
                print("Constructor signature:", inspect.signature(HippoRAG.__init__))
                raise
            hr.index(docs=docs)
            t1 = time.perf_counter()
            sol = hr.retrieve(queries=[it["question"]], num_to_retrieve=len(docs))
            sol = sol[0] if isinstance(sol, (list, tuple)) else sol
            t2 = time.perf_counter()
            pos = {d.strip(): i for i, d in enumerate(docs)}
            ranked = []
            for d in getattr(sol, "docs", []) or []:
                i = pos.get(str(d).strip())
                if i is not None and i not in ranked:
                    ranked.append(i)
            m = eval_ranking(ranked, supporting(g), len(docs))
            f.write(json.dumps({"id": it["id"], "ranked": ranked, "metrics": m,
                                "index_s": t1 - t0, "retrieve_s": t2 - t1}) + "\n")
            f.flush()
            shutil.rmtree(save, ignore_errors=True)
            print(f"{k + 1}/{len(items)} FR@5={m['FullRec@5']} index {t1 - t0:.1f}s", flush=True)

    rows = [json.loads(l) for l in open(path)]
    write_csv(os.path.join(out_dir, f"{a.dataset}_summary.csv"), [{
        "dataset": a.dataset, "queries": len(rows), "llm": a.llm, "embedding": a.embedding,
        **{m: round(mean([r["metrics"] for r in rows], m), 4) for m in REPORT},
        "mean_index_s": round(sum(r["index_s"] for r in rows) / len(rows), 2)}])


if __name__ == "__main__":
    main()
