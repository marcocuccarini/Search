#!/usr/bin/env python3
"""Search+ per-query cost, standalone (no repository, no pip install).

Needs only Python 3.8+ and a running Ollama server. The questions are in
subsample.json (100 random test questions per dataset, seed 0).

For every question it measures, with the order of the methods rotating from one
question to the next:
  aware     Search+ with query-aware KG:   LLM KG extraction (prompt of MineKG.py, with the
            question) + graph construction (BuildGraphsPerQuestion.py) + BFS / NSP reranking
  unaware   the same with the query-unaware KG (MineKG.py prompt without the question)
  listwise  list-wise LLM reranking (prompt of BaselineLLM.py, full passages, temperature 0)

    ollama serve
    python latency.py                          # all datasets, all methods
    python latency.py --model gemma4:e4b       # the tag you used in the paper
    python latency.py --mock                   # test the script without Ollama
    python latency.py --num-predict -1 --out results_nolimit --rerun-failed-from results
                                               # list-wise without the 1024 limit, re-calling the LLM
                                               # only where it failed or reached the limit

Resumable: stop with Ctrl+C and relaunch, finished questions are kept.
Outputs in results/: perquery_<dataset>.jsonl, summary.csv, latency_table.tex
"""
import argparse
import csv
import json
import os
import platform
import random
import statistics
import subprocess
import time
import urllib.request
from collections import defaultdict
from math import log

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
MODES = ("aware", "unaware", "listwise")
DATASETS = ("hotpotqa", "2wiki", "musique")
# BFS / NSP configurations used only for timing (time does not depend on the values)
BFS_CFG = {"top_k_bm25": 3, "max_depth": 2, "branch": 3, "alpha": 1.0}
NSP_CFG = {"ge_alpha": 0.5, "ge_hops": 2}


# ════════════════════════════ prompts (copied from the repository) ════════════════════════════
def kg_prompt(context, question=None):
    """MineKG.py, PromptLibrary.create_kg_prompt"""
    if question:
        return ("You are a Knowledge Graph engineer. "
                f"Based on the specific question: '{question}', analyze this text: '{context}'.\n\n"
                "Task: Extract entities and relationships that help answer the question.\n"
                "Return ONLY a JSON object with 'nodes' and 'edges'.")
    return ("You are a Knowledge Graph engineer. "
            f"Analyze the following text: '{context}'.\n\n"
            "Task: Extract all significant entities and their relationships to build a comprehensive Knowledge Graph.\n"
            "Return ONLY a JSON object with 'nodes' and 'edges'.")


def ranking_prompt(question, nodes, max_chars=None):
    """BaselineLLM.py, build_prompt (without the 200-character cut unless --listwise-max-chars)"""
    cut = (lambda t: t[:max_chars]) if max_chars else (lambda t: t)
    docs = "\n".join(f"[{i}] {n.get('title', '')} {cut(n.get('text', ''))}" for i, n in enumerate(nodes))
    return f"""
You are a document ranking system for information retrieval.

TASK:
Rank documents by relevance to the question.

STRICT OUTPUT RULES:
- ONLY output a JSON list of integers
- NO explanation
- NO extra text

Example:
[2, 0, 1, 3]

QUESTION:
{question}

DOCUMENTS:
{docs}
""".strip()


# ══════════════════════ graph construction (copied from BuildGraphsPerQuestion.py) ══════════════════════
def _entity_in_text(entity, text):
    if entity is None or text is None:
        return False
    return str(entity).lower() in str(text).lower()


def _get_triplet(kg_edge):
    if isinstance(kg_edge, list):
        try:
            s = kg_edge[0]
            r = kg_edge[1] if len(kg_edge) > 1 else "connected_to"
            o = kg_edge[2] if len(kg_edge) > 2 else ""
            return str(s), str(r), str(o)
        except Exception:
            return None
    elif isinstance(kg_edge, dict):
        s = kg_edge.get("source") or kg_edge.get("entity") or ""
        r = kg_edge.get("source_relationship") or kg_edge.get("relation") or "connected_to"
        o = kg_edge.get("target") or kg_edge.get("entity2") or ""
        return str(s), str(r), str(o)
    return None


def build_hybrid_pseudo_graph(item_id, item_data, kg_item):
    paragraphs = item_data.get("paragraphs", [])
    nodes = [{"idx": p.get("idx"), "title": p.get("title", ""), "text": p.get("paragraph_text", ""),
              "is_supporting": p.get("is_supporting", False)} for p in paragraphs]
    triplets, unique_entities = [], set()
    if kg_item:
        kg = kg_item.get("knowledge_graph", kg_item)
        raw_edges = kg.get("edges", []) if isinstance(kg, dict) else []
        for e in raw_edges or []:
            t = _get_triplet(e)
            if t and t[0] and t[2]:
                triplets.append(t)
                unique_entities.add(t[0])
                unique_entities.add(t[2])
    final_edges = []
    for s, r, o in triplets:
        paras_with_s = [n["idx"] for n in nodes if _entity_in_text(s, n["text"]) or _entity_in_text(s, n["title"])]
        paras_with_o = [n["idx"] for n in nodes if _entity_in_text(o, n["text"]) or _entity_in_text(o, n["title"])]
        for i in paras_with_s:
            for j in paras_with_o:
                if i == j:
                    continue
                final_edges.append({"src": i, "dst": j, "relation_type": "kg_bridge",
                                    "relation": r, "subject": s, "object": o})
    for entity in unique_entities:
        paras_with_ent = [n["idx"] for n in nodes if _entity_in_text(entity, n["text"]) or _entity_in_text(entity, n["title"])]
        for i in range(len(paras_with_ent)):
            for j in range(i + 1, len(paras_with_ent)):
                a, b = paras_with_ent[i], paras_with_ent[j]
                final_edges.append({"src": a, "dst": b, "relation_type": "entity_overlap", "shared_entity": entity})
                final_edges.append({"src": b, "dst": a, "relation_type": "entity_overlap", "shared_entity": entity})
    return {"id": item_id, "question": item_data.get("question", ""), "nodes": nodes, "edges": final_edges}


# ════════════════════════════ BFS / NSP (copied from TestExperiments.py) ════════════════════════════
def _adjacency(graph):
    adj_strength = defaultdict(lambda: defaultdict(float))
    for e in graph.get("edges", []):
        adj_strength[e["src"]][e["dst"]] += 1.0
        adj_strength[e["dst"]][e["src"]] += 1.0
    adj = defaultdict(list)
    for src in adj_strength:
        for dst, strength in adj_strength[src].items():
            adj[src].append((dst, strength))
    return adj


def compute_ge_embeddings(graph, base_scores, ge_alpha, ge_hops):
    n = len(graph["nodes"])
    adj = _adjacency(graph)
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


def bfs_retrieve_with_base(graph, base_scores, top_k_bm25, max_depth, branch, alpha):
    n = len(graph["nodes"])
    adj = _adjacency(graph)
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
                scored.append((j, alpha * (w / n) + (1 - alpha) * (base_scores[j] / max_s)))
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


def bm25(question, nodes, k1=1.5, b=0.75, epsilon=0.25):
    """Identical to rank_bm25.BM25Okapi, as used in the repository (same tokenisation)."""
    docs = [nd["title"].lower().split() + nd["text"].lower().split() for nd in nodes]
    N = len(docs)
    avgdl = sum(map(len, docs)) / N
    df = defaultdict(int)
    for d in docs:
        for t in set(d):
            df[t] += 1
    idf, neg = {}, []
    for t, f in df.items():
        idf[t] = log(N - f + 0.5) - log(f + 0.5)
        if idf[t] < 0:
            neg.append(t)
    eps = epsilon * (sum(idf.values()) / len(idf))
    for t in neg:
        idf[t] = eps
    tfs = []
    for d in docs:
        tf = defaultdict(int)
        for t in d:
            tf[t] += 1
        tfs.append(tf)
    out = []
    for d, tf in zip(docs, tfs):
        s = 0.0
        for t in question.lower().split():
            f = tf.get(t, 0)
            s += idf.get(t, 0) * (f * (k1 + 1) / (f + k1 * (1 - b + b * len(d) / avgdl)))
        out.append(s)
    return out


# ════════════════════════════════════ Ollama calls ════════════════════════════════════
def post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode())


class LLM:
    def __init__(self, model, host, num_ctx, mock, num_predict=1024):
        self.model, self.host, self.num_ctx, self.mock = model, host.rstrip("/"), num_ctx, mock
        self.num_predict = num_predict
        self.rng = random.Random(0)

    def _options(self, extra=None):
        o = dict(extra or {})
        if self.num_ctx:
            o["num_ctx"] = self.num_ctx
        return o

    def extract_kg(self, prompt):
        """Same call as MineKG.py: chat API, format='json', no temperature option."""
        t0 = time.perf_counter()
        try:
            if self.mock:
                time.sleep(0.002)
                j = {"message": {"content": json.dumps({"nodes": [], "edges": [{"source": "a", "target": "b"}]})},
                     "prompt_eval_count": len(prompt) // 4, "eval_count": 40}
            else:
                payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                           "format": "json", "stream": False}
                if self._options():
                    payload["options"] = self._options()
                j = post(self.host + "/api/chat", payload)
            raw, err = (j.get("message", {}) or {}).get("content", "").strip(), None
        except Exception as e:
            j, raw, err = {}, "", str(e)[:300]
        wall = time.perf_counter() - t0
        try:
            kg = json.loads(raw)
            ok = isinstance(kg, dict) and "error" not in kg
        except Exception:
            kg, ok = {"error": "JSON Parsing failed"}, False
        return kg, ok, wall, j, err

    def rank(self, prompt, n):
        """Same call as BaselineLLM.py: generate API, temperature 0, num_predict 1024."""
        t0 = time.perf_counter()
        try:
            if self.mock:
                time.sleep(0.002)
                perm = list(range(n))
                self.rng.shuffle(perm)
                j = {"response": json.dumps(perm), "prompt_eval_count": len(prompt) // 4, "eval_count": 3 * n}
            else:
                j = post(self.host + "/api/generate", {
                    "model": self.model, "prompt": prompt, "stream": False,
                    "options": self._options({"temperature": 0.0, "num_predict": self.num_predict})})
            text, err = j.get("response", "") or "", None
        except Exception as e:
            j, text, err = {}, "", str(e)[:300]
        wall = time.perf_counter() - t0
        try:
            out = json.loads(text.strip())
            valid = isinstance(out, list)
            complete = valid and sorted(int(x) for x in out) == list(range(n))
        except Exception:
            valid, complete = False, False
        return valid, complete, wall, j, err


def ollama_info(j):
    ns = 1e9
    thinking = j.get("thinking") or (j.get("message") or {}).get("thinking") or ""
    return {"prompt_tokens": j.get("prompt_eval_count"), "output_tokens": j.get("eval_count"),
            "thinking_chars": len(thinking), "done_reason": j.get("done_reason"),
            "ollama_load_s": (j.get("load_duration") or 0) / ns,
            "prompt_eval_s": (j.get("prompt_eval_duration") or 0) / ns,
            "generation_s": (j.get("eval_duration") or 0) / ns}


def median_time(fn, repeats):
    times, res = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        res = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), res


# ═══════════════════════════════════════ main ═══════════════════════════════════════
def run_one(llm, item, mode, listwise_max_chars):
    paragraphs = item["paragraphs"]
    base = {"id": item["id"], "mode": mode, "n_passages": len(paragraphs)}
    if mode == "listwise":
        nodes = [{"title": p["title"], "text": p["paragraph_text"]} for p in paragraphs]
        scores = bm25(item["question"], nodes)
        order = sorted(range(len(nodes)), key=lambda i: scores[i], reverse=True)   # retriever order
        prompt = ranking_prompt(item["question"], [nodes[i] for i in order], listwise_max_chars)
        valid, complete, wall, j, err = llm.rank(prompt, len(nodes))
        info = ollama_info(j)
        hit = bool(info["output_tokens"] and llm.num_predict > 0 and info["output_tokens"] >= llm.num_predict)
        return {**base, "llm_s": wall, "total_s": wall, "ok": valid, "complete_permutation": complete,
                "error": err, "hit_num_predict": hit, "num_predict": llm.num_predict, **info}
    context = " ".join(p["paragraph_text"] for p in paragraphs)
    prompt = kg_prompt(context, item["question"] if mode == "aware" else None)
    kg, ok, wall, j, err = llm.extract_kg(prompt)
    g_s, graph = median_time(lambda: build_hybrid_pseudo_graph(item["id"], item, {"knowledge_graph": kg}), 5)
    scores = bm25(item["question"], graph["nodes"])
    bfs_s, _ = median_time(lambda: bfs_retrieve_with_base(graph, scores, **BFS_CFG), 20)
    def nsp():
        ge = compute_ge_embeddings(graph, scores, **NSP_CFG)
        return sorted(range(len(ge)), key=lambda i: ge[i], reverse=True)
    nsp_s, _ = median_time(nsp, 20)
    edges = graph["edges"]
    return {**base, "llm_s": wall, "graph_build_s": g_s, "bfs_s": bfs_s, "nsp_s": nsp_s,
            "total_s": wall + g_s + bfs_s, "ok": ok, "error": err, "n_edges": len(edges),
            "n_kg_bridge": sum(e["relation_type"] == "kg_bridge" for e in edges), **ollama_info(j)}


def stats(vals):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return {"mean": "", "std": "", "median": "", "p95": ""}
    return {"mean": round(statistics.mean(v), 4), "std": round(statistics.pstdev(v), 4),
            "median": round(statistics.median(v), 4), "p95": round(v[min(len(v) - 1, int(0.95 * len(v)))], 4)}


def summarise(datasets, modes, model, gpu, nq):
    rows = []
    for ds in datasets:
        path = os.path.join(OUT, f"perquery_{ds}.jsonl")
        if not os.path.exists(path):
            continue
        recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        for mode in modes:
            sel = [r for r in recs if r["mode"] == mode]
            if not sel:
                continue
            row = {"dataset": ds, "mode": mode, "queries": len(sel), "model": model, "gpu": gpu}
            for key in ("llm_s", "total_s"):
                for s, v in stats([r.get(key) for r in sel]).items():
                    row[f"{key}_{s}"] = v
            for key in ("graph_build_s", "bfs_s", "nsp_s"):
                vals = [r[key] for r in sel if r.get(key) is not None]
                row[key.replace("_s", "_ms_mean")] = round(statistics.mean(vals) * 1000, 3) if vals else ""
            for key in ("prompt_tokens", "output_tokens", "n_edges"):
                vals = [r[key] for r in sel if r.get(key) is not None]
                row[f"{key}_mean"] = round(statistics.mean(vals), 1) if vals else ""
            gen = [r["output_tokens"] / r["generation_s"] for r in sel if r.get("output_tokens") and r.get("generation_s")]
            row["output_tokens_per_s"] = round(statistics.mean(gen), 1) if gen else ""
            row["rerun"] = sum(bool(r.get("rerun")) for r in sel)
            row["invalid_output_rate"] = round(1 - statistics.mean([bool(r["ok"]) for r in sel]), 4)
            row["call_error_rate"] = round(statistics.mean([bool(r.get("error")) for r in sel]), 4)
            if mode == "listwise":
                row["incomplete_permutation_rate"] = round(1 - statistics.mean([bool(r["complete_permutation"]) for r in sel]), 4)
                def hit(r):
                    if r.get("hit_num_predict") is not None:
                        return bool(r["hit_num_predict"])
                    return (r.get("output_tokens") or 0) >= 1024 and r.get("num_predict", 1024) > 0
                row["hit_num_predict_rate"] = round(statistics.mean([hit(r) for r in sel]), 4)
                bad = [r for r in sel if not r.get("complete_permutation")]
                row["failures_that_hit_limit"] = round(statistics.mean([hit(r) for r in bad]), 4) if bad else ""
                row["output_tokens_max"] = max((r.get("output_tokens") or 0) for r in sel)
            row["thinking_chars_mean"] = round(statistics.mean([r.get("thinking_chars") or 0 for r in sel]), 1)
            rows.append(row)
    if not rows:
        return
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(OUT, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)

    names = {"hotpotqa": "HotpotQA", "2wiki": "2WikiMHQA", "musique": "MuSiQue"}
    label = {"aware": "Search+ (query-aware)", "unaware": "Search+ (query-unaware)", "listwise": "List-wise LLM"}
    lines = ["% Generated by latency.py", "\\begin{table}[t]", "\\centering", "\\footnotesize",
             "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{llccccc}", "\\toprule",
             "\\textbf{Dataset} & \\textbf{Method} & \\textbf{LLM (s)} & \\textbf{Graph+rank (ms)} & "
             "\\textbf{Total (s)} & \\textbf{Out tok.} & \\textbf{Invalid}\\\\", "\\midrule"]
    shown = [ds for ds in ("hotpotqa", "2wiki", "musique") if any(r["dataset"] == ds for r in rows)]
    for k, ds in enumerate(shown):
        for j, r in enumerate([r for r in rows if r["dataset"] == ds]):
            gr = f"{r['graph_build_ms_mean'] + r['bfs_ms_mean']:.2f}" if r["graph_build_ms_mean"] != "" else "--"
            lines.append(f"{names[ds] if j == 0 else ''} & {label[r['mode']]} & "
                         f"{r['llm_s_mean']:.2f} $\\pm$ {r['llm_s_std']:.2f} & {gr} & {r['total_s_mean']:.2f} & "
                         f"{r['output_tokens_mean']} & {100 * r['invalid_output_rate']:.1f}\\%\\\\")
        if k < len(shown) - 1:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}",
              f"\\caption{{Per-query cost on {nq} random test questions per dataset ({model}, {gpu or 'GPU'}). "
              "LLM: time of the single LLM call (KG extraction for Search+, ranking for list-wise; mean "
              "$\\pm$ std). Graph+rank: graph construction plus BFS reranking. Invalid: LLM outputs that "
              "are not valid JSON (for list-wise, not a JSON list).}", "\\label{tab:latency}", "\\end{table}"]
    with open(os.path.join(OUT, "latency_table.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "=" * 100)
    for r in rows:
        extra = (f"  incomplete {r['incomplete_permutation_rate']}  hit_limit {r['hit_num_predict_rate']}"
                 f"  max_out_tok {r['output_tokens_max']}"
                 if r["mode"] == "listwise" else None) or \
            f"  build {r['graph_build_ms_mean']} ms  BFS {r['bfs_ms_mean']} ms  NSP {r['nsp_ms_mean']} ms"
        print(f"{r['dataset']:9s} {r['mode']:8s} LLM {r['llm_s_mean']} s (median {r['llm_s_median']}, p95 "
              f"{r['llm_s_p95']})  total {r['total_s_mean']} s  out_tok {r['output_tokens_mean']}  "
              f"invalid {r['invalid_output_rate']}{extra}")
    print("results/summary.csv  results/latency_table.tex")


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--n", type=int, default=100, help="questions per dataset (max 100)")
    ap.add_argument("--model", default="gemma4:latest", help="Ollama tag, as in MineKG.py")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--num-ctx", type=int, default=16384,
                    help="context window for ALL calls, so no prompt is truncated (0 = Ollama default)")
    ap.add_argument("--listwise-max-chars", type=int, default=0, help="0 = full passages; 200 = BaselineLLM.py")
    ap.add_argument("--num-predict", type=int, default=1024,
                    help="list-wise generation limit, thinking included (1024 = BaselineLLM.py; -1 = no limit)")
    ap.add_argument("--out", default="results", help="output folder (use another one for a second setting)")
    ap.add_argument("--rerun-failed-from", default=None, metavar="FOLDER",
                    help="e.g. results: copy all its measurements into --out and call the LLM again ONLY "
                         "for list-wise answers that failed or reached the 1024-token limit")
    ap.add_argument("--warmup", type=int, default=3, help="untimed calls per method before measuring")
    ap.add_argument("--mock", action="store_true", help="fake LLM, to test the script without Ollama")
    a = ap.parse_args()

    OUT = os.path.join(HERE, a.out)
    data = json.load(open(os.path.join(HERE, "subsample.json"), encoding="utf-8"))["datasets"]
    os.makedirs(OUT, exist_ok=True)
    gpu = ""
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
    except Exception:
        pass
    llm = LLM(a.model, a.host, a.num_ctx or None, a.mock, a.num_predict)
    if not a.mock:
        try:
            post(a.host.rstrip("/") + "/api/show", {"model": a.model})
        except Exception as e:
            raise SystemExit(f"Cannot reach Ollama at {a.host} with model '{a.model}': {e}\n"
                             f"Start it with 'ollama serve' and check the tag with 'ollama list'.")
    print(f"model={a.model}  gpu={gpu or 'unknown'}  cpu={platform.processor() or platform.machine()}  num_ctx={a.num_ctx or 'default'}")

    first = data[a.datasets[0]][0]
    print(f"warm-up ({a.warmup} calls per method, not timed)...")
    for _ in range(a.warmup):
        for mode in a.modes:
            run_one(llm, first, mode, a.listwise_max_chars or None)

    if a.rerun_failed_from:
        if a.num_predict == 1024:
            raise SystemExit("--rerun-failed-from needs a different --num-predict (e.g. -1 = no limit)")
        src = os.path.join(HERE, a.rerun_failed_from)
        os.makedirs(OUT, exist_ok=True)
        for ds in a.datasets:
            sp, dp = os.path.join(src, f"perquery_{ds}.jsonl"), os.path.join(OUT, f"perquery_{ds}.jsonl")
            if not os.path.exists(sp) or os.path.exists(dp):
                continue
            kept = redo = 0
            with open(dp, "w", encoding="utf-8") as f:
                for l in open(sp, encoding="utf-8"):
                    if not l.strip():
                        continue
                    r = json.loads(l)
                    if r["mode"] == "listwise" and (not r.get("complete_permutation")
                                                   or (r.get("output_tokens") or 0) >= 1024):
                        redo += 1          # not copied: measured again below without the limit
                        continue
                    f.write(json.dumps({**r, "rerun": False}) + "\n")
                    kept += 1
            print(f"{ds}: copied {kept} measurements from {a.rerun_failed_from}/, {redo} list-wise answers to redo")
        a.modes = ["listwise"]
        a.warmup = min(a.warmup, 1)

    t_start = time.time()
    for ds in a.datasets:
        items = data[ds][:a.n]
        path = os.path.join(OUT, f"perquery_{ds}.jsonl")
        done = set()
        if os.path.exists(path):
            for l in open(path, encoding="utf-8"):
                if l.strip():
                    r = json.loads(l)
                    done.add((r["id"], r["mode"]))
        todo = sum((item["id"], m) not in done for item in items for m in a.modes)
        if todo == 0:
            print(f"  {ds}: all {len(items)} questions already measured in {a.out}/ (reused, nothing new "
                  f"is measured). To measure again use a new --out folder.")
            continue
        print(f"  {ds}: {todo} measurements to do ({len(done)} reused from {a.out}/)")
        with open(path, "a", encoding="utf-8") as f:
            for k, item in enumerate(items):
                if all((item["id"], m) in done for m in a.modes):
                    continue
                modes = list(a.modes)
                order = modes[k % len(modes):] + modes[:k % len(modes)]   # rotate the order
                for mode in order:
                    if (item["id"], mode) in done:
                        continue
                    rec = run_one(llm, item, mode, a.listwise_max_chars or None)
                    rec["dataset"] = ds
                    if a.rerun_failed_from:
                        rec["rerun"] = True
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    if rec.get("error"):
                        print(f"  [error] {ds} {item['id']} {mode}: {rec['error']}")
                el = time.time() - t_start
                print(f"  {ds:9s} {k + 1:3d}/{len(items)}   elapsed {el / 60:5.1f} min", flush=True)
    summarise(a.datasets, MODES if a.rerun_failed_from else a.modes, a.model, gpu, a.n)


if __name__ == "__main__":
    main()
