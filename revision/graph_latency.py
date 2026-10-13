"""Per-query cost of Search+ vs list-wise LLM reranking (reviewer point 8).

Measured modes (on the SAME random test questions):
  aware     Search+ with query-aware KG: LLM extraction + graph build + BFS/NSP rerank
  unaware   same with the query-unaware KG
  listwise  list-wise LLM reranking. By default (--listwise-source reuse) its cost is
            NOT measured again: it is read, for the same sampled questions, from the
            per-question log of llm_rerank.py (full passages, same protocol as the
            accuracy results). Run llm_rerank.py first. --listwise-source live
            instead calls the LLM here with BaselineLLM.py's prompt (200-char cut).

For the two Search+ modes it measures:
  1. KG extraction: one Ollama call with exactly the prompt and call of MineKG.py
     (PromptLibrary.create_kg_prompt, client.chat(..., format='json')), with and
     without the question;
  2. graph construction: BuildGraphsPerQuestion.build_hybrid_pseudo_graph on the
     extracted KG.
  3. reranking: BFS and NSP (repo functions) on the resulting graph.
The order of the modes rotates question by question so that GPU warm-up or
thermal drift does not favour one of them.

    ollama serve
    python revision/graph_latency.py --n 100                       # 100 questions per dataset
    python revision/graph_latency.py --n 100 --model gemma4:e4b    # model tag used in the paper

Resumable: stop with Ctrl+C and relaunch, finished questions are kept.
Outputs (revision_results/latency/):
  perquery_<ds>.jsonl  one line per question x mode
  summary.csv          mean / median / p95 per dataset x mode, tokens, JSON failures
  latency_table.tex    LaTeX table for the paper
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, write_csv  # noqa: E402
from core import (DATASETS, REPO, bfs_retrieve_with_base, compute_ge_embeddings,  # noqa: E402
                  load_items, load_paper_graphs)

sys.path.insert(0, REPO)
from BuildGraphsPerQuestion import build_hybrid_pseudo_graph  # noqa: E402

LAT_DIR = os.path.join(OUT, "latency")
MODES = ("aware", "unaware", "listwise")
BFS_CFG = {"top_k_bm25": 3, "max_depth": 2, "branch": 3, "alpha": 1.0}
NSP_CFG = {"ge_alpha": 0.5, "ge_hops": 2}
NS = 1e9


def get(obj, key, default=None):
    """Field access that works for dicts and for ollama response objects."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class Extractor:
    """Same prompt and same Ollama call as MineKG.py."""

    def __init__(self, model, host, mock=False, temperature=None):
        self.model, self.mock, self.temperature = model, mock, temperature
        if not mock:
            import ollama
            from MineKG import PromptLibrary
            self.client = ollama.Client(host=host)
            self.prompt = PromptLibrary.create_kg_prompt
        else:
            self.prompt = lambda context, question=None: (question or "") + context

    def __call__(self, item, use_question):
        context = " ".join(p["paragraph_text"] for p in item.get("paragraphs", []))
        prompt = self.prompt(context, item["question"] if use_question else None)
        t0 = time.perf_counter()
        if self.mock:
            time.sleep(0.002)
            ents = [p["title"] for p in item["paragraphs"][:4]]
            raw = json.dumps({"nodes": ents, "edges": [{"source": ents[0], "target": ents[1],
                                                        "relation": "r"}]})
            resp = {"prompt_eval_count": len(prompt) // 4, "eval_count": len(raw) // 4,
                    "total_duration": 0, "load_duration": 0, "prompt_eval_duration": 0,
                    "eval_duration": 0}
        else:
            kwargs = dict(model=self.model, messages=[{"role": "user", "content": prompt}],
                          format="json", stream=False)
            if self.temperature is not None:
                kwargs["options"] = {"temperature": self.temperature}
            resp = self.client.chat(**kwargs)
            raw = (get(get(resp, "message", {}), "content", "") or "").strip()
        wall = time.perf_counter() - t0
        try:
            kg = json.loads(raw)
            ok = isinstance(kg, dict) and "error" not in kg
        except Exception:
            kg, ok = {"error": "JSON Parsing failed", "raw": raw}, False
        return {"kg": kg, "ok": ok, "llm_s": wall,
                "prompt_tokens": get(resp, "prompt_eval_count"),
                "output_tokens": get(resp, "eval_count"),
                "ollama_total_s": (get(resp, "total_duration") or 0) / NS,
                "ollama_load_s": (get(resp, "load_duration") or 0) / NS,
                "prompt_eval_s": (get(resp, "prompt_eval_duration") or 0) / NS,
                "generation_s": (get(resp, "eval_duration") or 0) / NS}


class ListwiseRanker:
    """Same prompt, parser and Ollama call as BaselineLLM.py."""

    def __init__(self, model, host, mock=False):
        self.model, self.mock = model, mock
        self.url = host.rstrip("/") + "/api/generate"
        if not mock:
            from BaselineLLM import build_prompt, parse_rank
        else:
            def build_prompt(q, nodes):
                return q + "".join(f"[{i}]" for i in range(len(nodes)))

            def parse_rank(text, n):
                return list(range(n))
        self.build_prompt, self.parse_rank = build_prompt, parse_rank

    def __call__(self, question, nodes):
        prompt = self.build_prompt(question, nodes)
        n = len(nodes)
        t0 = time.perf_counter()
        if self.mock:
            time.sleep(0.002)
            j = {"response": json.dumps(list(range(n))[::-1]), "prompt_eval_count": len(prompt) // 4,
                 "eval_count": 2 * n}
        else:
            import requests
            r = requests.post(self.url, timeout=600, json={
                "model": self.model, "prompt": prompt, "stream": False,
                "options": {"temperature": 0.0, "num_predict": 1024}})
            r.raise_for_status()
            j = r.json()
        wall = time.perf_counter() - t0
        text = j.get("response", "") or ""
        # validity, judged like parse_rank: a JSON list; complete = every index exactly once
        try:
            out = json.loads(text.strip())
            valid = isinstance(out, list)
            ids = [int(x) for x in out] if valid else []
            complete = valid and sorted(ids) == list(range(n))
        except Exception:
            valid, complete = False, False
        return {"ok": valid, "complete_permutation": complete, "llm_s": wall,
                "prompt_tokens": j.get("prompt_eval_count"), "output_tokens": j.get("eval_count"),
                "ollama_total_s": (j.get("total_duration") or 0) / NS,
                "ollama_load_s": (j.get("load_duration") or 0) / NS,
                "prompt_eval_s": (j.get("prompt_eval_duration") or 0) / NS,
                "generation_s": (j.get("eval_duration") or 0) / NS}


def rerank_time(graph, scores, repeats=20):
    """Median wall time of BFS and NSP (repo functions, both edge types)."""
    tb, tn = [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        bfs_retrieve_with_base(graph, scores, **BFS_CFG, graph_type="both")
        tb.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        ge = compute_ge_embeddings(graph, scores, **NSP_CFG, graph_type="both")
        sorted(range(len(ge)), key=lambda i: ge[i], reverse=True)
        tn.append(time.perf_counter() - t0)
    return float(np.median(tb)), float(np.median(tn))


def bm25(question, graph):
    from rank_bm25 import BM25Okapi
    corpus = [nd["title"].lower().split() + nd["text"].lower().split() for nd in graph["nodes"]]
    return BM25Okapi(corpus).get_scores(question.lower().split()).tolist()


def build_time(item, kg, repeats=5):
    """Median wall time of the repo's graph builder (fast, so repeated)."""
    times, graph = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        graph = build_hybrid_pseudo_graph(item["id"], item, {"knowledge_graph": kg})
        times.append(time.perf_counter() - t0)
    edges = graph["edges"]
    return float(np.median(times)), graph, {
        "n_edges": len(edges),
        "n_kg_bridge": sum(e["relation_type"] == "kg_bridge" for e in edges),
        "n_entity_overlap": sum(e["relation_type"] == "entity_overlap" for e in edges)}


def stats(x):
    x = np.asarray([v for v in x if v is not None], float)
    if len(x) == 0:
        return {"mean": "", "std": "", "median": "", "p95": ""}
    return {"mean": round(float(x.mean()), 4), "std": round(float(x.std()), 4),
            "median": round(float(np.median(x)), 4), "p95": round(float(np.percentile(x, 95)), 4)}


def reuse_listwise(ds, retriever, ids):
    """List-wise records for the sampled questions, from llm_rerank.py's log."""
    path = os.path.join(OUT, "llm_rerank", f"{ds}_{retriever}.jsonl")
    if not os.path.exists(path):
        print(f"[note] {os.path.relpath(path, os.path.dirname(OUT))} not found: run "
              f"llm_rerank.py --dataset {ds} --retriever {retriever} first (list-wise skipped)")
        return []
    by_id = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            by_id[r["id"]] = r
    out = []
    for qid in ids:
        r = by_id.get(qid)
        if r is None:
            continue
        out.append({"id": qid, "dataset": ds, "mode": "listwise", "llm_s": r["llm_s"],
                    "total_s": r["llm_s"], "prompt_tokens": r.get("prompt_tokens"),
                    "output_tokens": r.get("output_tokens"),
                    "ok": r["status"] in ("valid", "incomplete"),
                    "complete_permutation": r["status"] == "valid", "source": "llm_rerank.py"})
    missing = len(ids) - len(out)
    if missing:
        print(f"[note] {ds}: {missing} sampled questions not yet in llm_rerank.py's log")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--n", type=int, default=100, help="questions per dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="gemma4:latest", help="same tag as MineKG.py")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--temperature", type=float, default=None,
                    help="leave unset to reproduce MineKG.py (no temperature option)")
    ap.add_argument("--warmup", type=int, default=3, help="untimed calls before measuring")
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--listwise-source", default="reuse", choices=["reuse", "live"],
                    help="reuse = read list-wise timings from llm_rerank.py logs (no new LLM calls)")
    ap.add_argument("--retriever", default="BGE-large", help="llm_rerank.py run to reuse")
    ap.add_argument("--mock", action="store_true", help="fake LLM, to test the script")
    a = ap.parse_args()

    os.makedirs(LAT_DIR, exist_ok=True)
    ex = Extractor(a.model, a.host, a.mock, a.temperature)
    lw = ListwiseRanker(a.model, a.host, a.mock)
    gpu = ""
    try:
        import subprocess
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        pass
    print(f"model={a.model}  gpu={gpu or 'unknown'}  cpu={platform.processor()}")

    first = load_items(a.datasets[0], "test")[0]
    first_graph = load_paper_graphs(a.datasets[0], "test")[first["id"]]
    live_modes = [m for m in a.modes if not (m == "listwise" and a.listwise_source == "reuse")]
    print(f"warm-up ({a.warmup} calls per mode, not timed)...")
    for _ in range(a.warmup):
        ex(first, True)
        if "listwise" in live_modes:
            lw(first["question"], first_graph["nodes"])

    samples = {}
    for ds in a.datasets:
        items = load_items(ds, "test")
        graphs = load_paper_graphs(ds, "test")
        items = [it for it in items if it["id"] in graphs]
        sample = random.Random(a.seed).sample(items, min(a.n, len(items)))
        samples[ds] = [it["id"] for it in sample]
        path = os.path.join(LAT_DIR, f"perquery_{ds}.jsonl")
        done = set()
        if os.path.exists(path):
            done = {(json.loads(l)["id"], json.loads(l)["mode"]) for l in open(path) if l.strip()}
        with open(path, "a") as f:
            for k, it in enumerate(sample):
                modes = list(live_modes)
                order = modes[k % len(modes):] + modes[:k % len(modes)]
                for mode in order:
                    if (it["id"], mode) in done:
                        continue
                    base = {"id": it["id"], "dataset": ds, "mode": mode,
                            "n_passages": len(it["paragraphs"])}
                    if mode == "listwise":
                        r = lw(it["question"], graphs[it["id"]]["nodes"])
                        rec = {**base, **r, "total_s": r["llm_s"]}
                    else:
                        r = ex(it, use_question=(mode == "aware"))
                        g_s, graph, g_info = build_time(it, r.pop("kg"))
                        t_bfs, t_nsp = rerank_time(graph, bm25(it["question"], graph))
                        rec = {**base, **r, "graph_build_s": g_s, "bfs_s": t_bfs, "nsp_s": t_nsp,
                               "total_s": r["llm_s"] + g_s + t_bfs, **g_info}
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                print(f"  {ds} {k + 1}/{len(sample)}", flush=True)

    # ---------------- summary
    rows = []
    for ds in a.datasets:
        path = os.path.join(LAT_DIR, f"perquery_{ds}.jsonl")
        if not os.path.exists(path):
            continue
        sample_ids = set(samples.get(ds, []))
        recs = [json.loads(l) for l in open(path) if l.strip()]
        recs = [r for r in recs if r["id"] in sample_ids]
        if "listwise" in a.modes and a.listwise_source == "reuse":
            recs = [r for r in recs if r["mode"] != "listwise"] + reuse_listwise(ds, a.retriever, samples[ds])
        for mode in MODES:
            sel = [r for r in recs if r["mode"] == mode]
            if not sel:
                continue
            row = {"dataset": ds, "mode": mode, "queries": len(sel), "model": a.model, "gpu": gpu}
            for key, label in (("llm_s", "llm_s"), ("total_s", "total_s")):
                for s, v in stats([r.get(key) for r in sel]).items():
                    row[f"{label}_{s}"] = v
            for key in ("graph_build_s", "bfs_s", "nsp_s"):
                vals = [r[key] for r in sel if r.get(key) is not None]
                row[key.replace("_s", "_ms_mean")] = round(float(np.mean(vals)) * 1000, 3) if vals else ""
            for key in ("prompt_tokens", "output_tokens", "n_edges"):
                vals = [r[key] for r in sel if r.get(key) is not None]
                row[f"{key}_mean"] = round(float(np.mean(vals)), 1) if vals else ""
            gen = [r["output_tokens"] / r["generation_s"] for r in sel
                   if r.get("output_tokens") and r.get("generation_s")]
            row["output_tokens_per_s"] = round(float(np.mean(gen)), 1) if gen else ""
            row["invalid_output_rate"] = round(1 - float(np.mean([r["ok"] for r in sel])), 4)
            if mode == "listwise":
                row["incomplete_permutation_rate"] = round(
                    1 - float(np.mean([r["complete_permutation"] for r in sel])), 4)
            rows.append(row)
    write_csv(os.path.join(LAT_DIR, "summary.csv"), rows)

    names = {"hotpotqa": "HotpotQA", "2wiki": "2WikiMHQA", "musique": "MuSiQue"}
    lines = ["% Generated by revision/graph_latency.py",
             "\\begin{table}[t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{llccccc}", "\\toprule",
             "\\textbf{Dataset} & \\textbf{Method} & \\textbf{LLM (s)} & \\textbf{Graph+rank (ms)} & "
             "\\textbf{Total (s)} & \\textbf{Out tok.} & \\textbf{Invalid}\\\\", "\\midrule"]
    label = {"aware": "Search+ (query-aware)", "unaware": "Search+ (query-unaware)",
             "listwise": "List-wise LLM"}
    for ds in ("hotpotqa", "2wiki", "musique"):
        for j, r in enumerate([r for r in rows if r["dataset"] == ds]):
            gr = (f"{r['graph_build_ms_mean'] + r['bfs_ms_mean']:.2f}"
                  if r["graph_build_ms_mean"] != "" else "--")
            lines.append(f"{names[ds] if j == 0 else ''} & {label[r['mode']]} & "
                         f"{r['llm_s_mean']:.2f} $\\pm$ {r['llm_s_std']:.2f} & {gr} & "
                         f"{r['total_s_mean']:.2f} & {r['output_tokens_mean']} & "
                         f"{100 * r['invalid_output_rate']:.1f}\\%\\\\")
        if any(r["dataset"] == ds for r in rows) and ds != "musique":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}",
              f"\\caption{{Per-query graph construction cost on {max(r['queries'] for r in rows) if rows else '?'} "
              f"random test questions per dataset ({a.model}, {gpu or 'GPU'}). LLM: time of the single "
              "LLM call (KG extraction for Search+, ranking for list-wise; mean $\\pm$ std). "
              "Graph+rank: graph construction plus BFS reranking. Invalid: LLM outputs that are not "
              "valid JSON (for list-wise, not a JSON list).}", "\\label{tab:latency}", "\\end{table}"]
    with open(os.path.join(LAT_DIR, "latency_table.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  -> revision_results/latency/latency_table.tex")
    for r in rows:
        print(f"{r['dataset']:9s} {r['mode']:8s} LLM {r['llm_s_mean']}s "
              f"(median {r['llm_s_median']}, p95 {r['llm_s_p95']})  total {r['total_s_mean']}s  "
              f"build {r['graph_build_ms_mean']} ms  BFS {r['bfs_ms_mean']} ms  NSP {r['nsp_ms_mean']} ms  "
              f"out_tokens {r['output_tokens_mean']}  invalid {r['invalid_output_rate']}"
              + (f"  incomplete {r['incomplete_permutation_rate']}" if r['mode'] == 'listwise' else ""))


if __name__ == "__main__":
    main()
