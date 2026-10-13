"""List-wise LLM reranking evaluated with the SAME protocol as Search+.

Protocol (identical to Search+ in run_experiments.py):
  * all test questions, all candidate passages of each question;
  * the LLM starts from the same retriever (default BGE-large): passages are
    given in retriever order;
  * failures are COUNTED, not dropped: when the LLM call fails or its output is
    not a valid JSON list, the question keeps the retriever ranking, exactly as
    Search+ falls back to the retriever ranking when KG extraction fails (empty
    graph). If the list is valid but incomplete, the missing passages are
    appended in retriever order;
  * same metrics (eval_ranking from TestExperiments.py) and paired bootstrap
    against Search+ BFS / NSP on the same questions.

Prompt text, model, temperature 0 and num_predict 1024 as in BaselineLLM.py (with
thinking models the limit also counts thinking tokens: use --num-predict -1 to remove
it; hit_num_predict_rate in the summary shows how often it truncated the output), but
with full passages (BaselineLLM.py cut them to 200 characters) and an explicit
num_ctx, so Ollama never truncates long prompts silently.

    ollama serve
    python revision/llm_rerank.py --dataset musique --retriever BGE-large
    python revision/llm_rerank.py --dataset musique --retriever BGE-large --max-chars 200   # old setting
    # afterwards, remove the 1024-token limit by re-calling the LLM ONLY where it failed or hit the limit:
    python revision/llm_rerank.py --dataset musique --retriever BGE-large --num-predict -1 --rerun-failed

Resumable (Ctrl+C and relaunch). Outputs in revision_results/llm_rerank/:
  <ds>_<retriever>[_cut200].jsonl   per question (ranking, status, raw output, time)
  summary.csv                       one row per run
  significance.csv                  Search+ BFS / NSP vs LLM, paired bootstrap
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, ensure_hashseed, paired_bootstrap  # noqa: E402
from core import eval_ranking, load_items, load_paper_graphs, rank, supporting  # noqa: E402
from scores import get_scores  # noqa: E402

DIR = os.path.join(OUT, "llm_rerank")
REPORT = ["Recall@3", "Recall@5", "Recall@10", "FullRec@3", "FullRec@5", "FullRec@10", "NDCG@10", "MRR"]
NS = 1e9


def build_prompt(question, nodes, max_chars=None):
    """Prompt text identical to BaselineLLM.build_prompt, without the 200-char cut."""
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


def parse(text, n):
    """-> (order, status). status: valid | incomplete | invalid.
    Strict JSON as in BaselineLLM.parse_rank. incomplete = a valid list that
    misses, repeats or invents indices (the usable indices are kept)."""
    try:
        out = json.loads(text.strip())
    except Exception:
        return [], "invalid"
    if not isinstance(out, list):
        return [], "invalid"
    order, seen, clean = [], set(), True
    for x in out:
        try:
            x = int(x)
        except Exception:
            clean = False
            continue
        if 0 <= x < n and x not in seen:
            seen.add(x)
            order.append(x)
        else:
            clean = False
    return order, ("valid" if clean and len(order) == n else "incomplete")


class LLM:
    def __init__(self, model, host, num_ctx, num_predict=1024, mock=False):
        self.model, self.url, self.num_ctx, self.mock = model, host.rstrip("/") + "/api/generate", num_ctx, mock
        self.num_predict = num_predict
        self.rng = np.random.default_rng(0)

    def __call__(self, prompt, n):
        t0 = time.perf_counter()
        try:
            if self.mock:
                time.sleep(0.001)
                r = self.rng.random()
                text = ("[0, 1" if r < 0.05 else
                        json.dumps([int(x) for x in self.rng.permutation(n)][:n - 1]) if r < 0.1 else
                        json.dumps([int(x) for x in self.rng.permutation(n)]))
                j = {"response": text, "prompt_eval_count": len(prompt) // 4, "eval_count": 3 * n}
            else:
                import requests
                resp = requests.post(self.url, timeout=600, json={
                    "model": self.model, "prompt": prompt, "stream": False,
                    "options": {"temperature": 0.0, "num_predict": self.num_predict, "num_ctx": self.num_ctx}})
                resp.raise_for_status()
                j = resp.json()
            error = None
        except Exception as e:  # counted as a failure, like MineKG.py does
            j, error = {}, str(e)[:300]
        out_tok = j.get("eval_count")
        return j.get("response", "") or "", {
            "llm_s": time.perf_counter() - t0, "error": error,
            "prompt_tokens": j.get("prompt_eval_count"), "output_tokens": out_tok,
            "thinking_chars": len(j.get("thinking") or ""),
            "hit_num_predict": bool(out_tok and self.num_predict > 0 and out_tok >= self.num_predict),
            "done_reason": j.get("done_reason")}


def main():
    ensure_hashseed()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["2wiki", "hotpotqa", "musique"])
    ap.add_argument("--retriever", default="BGE-large")
    ap.add_argument("--max-chars", type=int, default=0, help="0 = full passages; 200 = BaselineLLM.py")
    ap.add_argument("--model", default="gemma4:latest")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--num-predict", type=int, default=1024,
                    help="generation limit, thinking included (1024 = BaselineLLM.py; -1 = no limit)")
    ap.add_argument("--rerun-failed", action="store_true",
                    help="with --num-predict -1: start from the existing 1024-limit run and call the LLM "
                         "again ONLY for questions that failed or reached the limit; all others are copied")
    ap.add_argument("--mock", action="store_true", help="fake LLM, to test the script")
    a = ap.parse_args()
    if a.rerun_failed and a.num_predict == 1024:
        ap.error("--rerun-failed needs a different --num-predict (e.g. -1 = no limit)")

    items = load_items(a.dataset, "test")
    graphs = load_paper_graphs(a.dataset, "test")
    items = [it for it in items if it["id"] in graphs]
    scores = get_scores(a.retriever, a.dataset, "test", items, graphs)

    tag = f"{a.dataset}_{a.retriever}" + (f"_cut{a.max_chars}" if a.max_chars else "") + \
        ("" if a.num_predict == 1024 else f"_np{'unlimited' if a.num_predict < 0 else a.num_predict}")
    os.makedirs(DIR, exist_ok=True)
    path = os.path.join(DIR, tag + ".jsonl")
    done = {json.loads(l)["id"] for l in open(path) if l.strip()} if os.path.exists(path) else set()
    llm = LLM(a.model, a.host, a.num_ctx, a.num_predict, a.mock)

    if a.rerun_failed:
        base_tag = f"{a.dataset}_{a.retriever}" + (f"_cut{a.max_chars}" if a.max_chars else "")
        base_path = os.path.join(DIR, base_tag + ".jsonl")
        if not os.path.exists(base_path):
            raise SystemExit(f"--rerun-failed: {base_path} not found (run without the option first)")
        base = {json.loads(l)["id"]: json.loads(l) for l in open(base_path) if l.strip()}
        copied = 0
        with open(path, "a") as f:
            for it in items:
                r = base.get(it["id"])
                if r is None or it["id"] in done:
                    continue
                hit = (r.get("output_tokens") or 0) >= 1024
                if r["status"] == "valid" and not hit:
                    f.write(json.dumps({**r, "rerun": False}) + "\n")   # unchanged without the limit
                    done.add(it["id"])
                    copied += 1
        print(f"{base_tag}: copied {copied} questions that did not fail nor reach the limit; "
              f"the LLM is called again for the others")
    print(f"{tag}: {len(done)} done, {len(items) - len(done)} to go")

    with open(path, "a") as f:
        for q, it in enumerate(items):
            if it["id"] in done:
                continue
            g, s = graphs[it["id"]], scores[it["id"]]
            n = len(s)
            retr = rank("none", g, s, {}, "both")                 # retriever ranking
            prompt = build_prompt(it["question"], [g["nodes"][i] for i in retr], a.max_chars or None)
            text, info = llm(prompt, n)
            local, status = parse(text, n)
            if info["error"]:
                status = "error"
            final = [retr[i] for i in local]                      # LLM positions -> passage ids
            final += [i for i in retr if i not in final]          # failure / missing: retriever order
            supp = supporting(g)
            f.write(json.dumps({"id": it["id"], "status": status, "final": final, "retriever": retr,
                                "metrics": eval_ranking(final, supp, n),
                                "retriever_metrics": eval_ranking(retr, supp, n),
                                "raw": text[:500], "rerun": bool(a.rerun_failed), **info}) + "\n")
            f.flush()
            if (q + 1) % 50 == 0:
                print(f"  {q + 1}/{len(items)}", flush=True)

    # ---------------- summary (ALL test questions, failures included)
    by_id = {json.loads(l)["id"]: json.loads(l) for l in open(path) if l.strip()}
    recs = [by_id[it["id"]] for it in items if it["id"] in by_id]
    if len(recs) < len(items):
        print(f"[note] {len(items) - len(recs)} questions still missing: relaunch to finish")
    st = [r["status"] for r in recs]
    row = {"run": tag, "dataset": a.dataset, "retriever": a.retriever, "model": a.model,
           "passages": "full" if not a.max_chars else f"first {a.max_chars} chars",
           "queries": len(recs), "test_queries": len(items)}
    for m in REPORT:
        row[f"retriever_{m}"] = round(float(np.mean([r["retriever_metrics"][m] for r in recs])), 4)
        row[f"llm_{m}"] = round(float(np.mean([r["metrics"][m] for r in recs])), 4)
    row.update({k + "_rate": round(st.count(k) / len(st), 4) for k in ("valid", "incomplete", "invalid", "error")})
    row["failure_rate"] = round(row["invalid_rate"] + row["error_rate"], 4)
    row["num_predict"] = a.num_predict
    row["hit_num_predict_rate"] = round(float(np.mean([bool(r.get("hit_num_predict")) for r in recs])), 4)
    bad = [r for r in recs if r["status"] in ("invalid", "incomplete")]
    row["failures_that_hit_limit"] = round(float(np.mean([bool(r.get("hit_num_predict")) for r in bad])), 4) if bad else ""
    row["rerun_questions"] = sum(bool(r.get("rerun")) for r in recs)
    row.update({"llm_s_mean": round(float(np.mean([r["llm_s"] for r in recs])), 3),
                "prompt_tokens_mean": round(float(np.mean([r["prompt_tokens"] or 0 for r in recs])), 1),
                "prompt_tokens_max": int(max(r["prompt_tokens"] or 0 for r in recs))})
    if row["prompt_tokens_max"] >= a.num_ctx - 50:
        print(f"[WARNING] some prompts reach num_ctx={a.num_ctx}: raise --num-ctx and rerun them")

    # ---------------- Search+ on the same questions (from run_experiments.py)
    sig = []
    perq_path = os.path.join(OUT, "perquery", f"{a.dataset}_{a.retriever}.json")
    if os.path.exists(perq_path):
        perq = json.load(open(perq_path))
        pos = {qid: k for k, qid in enumerate(perq["qids"])}
        idx = [pos[r["id"]] for r in recs]
        for mode, name in (("bfs", "BFS"), ("ge", "NSP")):
            key = f"aware|{mode}|both"
            for m in ("FullRec@5", "Recall@5"):
                ours = np.array(perq[key][m])[idx]
                llm_v = [r["metrics"][m] for r in recs]
                row[f"searchplus_{name}_{m}"] = round(float(ours.mean()), 4)
                d, lo, hi, p = paired_bootstrap(ours, llm_v)
                sig.append({"run": tag, "A": f"Search+ {name}", "B": "LLM list-wise", "metric": m,
                            "mean_A": round(float(ours.mean()), 4), "mean_B": round(float(np.mean(llm_v)), 4),
                            "diff": round(d, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                            "p_value": round(p, 4), "significant": int(p < 0.05)})
    else:
        print(f"[note] no {os.path.relpath(perq_path, os.path.dirname(OUT))}: run run_experiments.py "
              f"--methods {a.retriever} to add the Search+ comparison")

    for name, new in (("summary.csv", [row]), ("significance.csv", sig)):
        if not new:
            continue
        p = os.path.join(DIR, name)
        rows = [r for r in csv.DictReader(open(p))] if os.path.exists(p) else []
        rows = [r for r in rows if r["run"] != tag] + new
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, keys)
            w.writeheader()
            w.writerows(rows)

    show = ["queries", "retriever_FullRec@5", "llm_FullRec@5", "retriever_Recall@5", "llm_Recall@5",
            "valid_rate", "incomplete_rate", "invalid_rate", "error_rate", "hit_num_predict_rate",
            "failures_that_hit_limit", "llm_s_mean", "prompt_tokens_mean"]
    show += [k for k in row if k.startswith("searchplus_")]
    print(json.dumps({k: row[k] for k in show}, indent=1))
    for s_ in sig:
        print(f"  {s_['A']} - LLM ({s_['metric']}): {s_['diff']:+.4f}  p={s_['p_value']}")
    print(f"  -> revision_results/llm_rerank/summary.csv")


if __name__ == "__main__":
    main()
