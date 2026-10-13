#!/usr/bin/env python3
r"""HippoRAG 2 on the same 100 questions per dataset as latency.py (subsample.json).

Same setting as Search+: each question's own candidate passages (10 or 20) are the
corpus; HippoRAG 2 indexes them (OpenIE with the LLM + embeddings), then ranks all
of them for the question. Same metrics as the paper (R@3, R@5, FR@3, FR@5), plus the
time of indexing and retrieval per question.

REQUIREMENTS (works on Windows: vLLM is not needed, the LLM is served by Ollama)
    python -m venv hippo              # separate environment, Python 3.10-3.12
    hippo\Scripts\activate            # Linux: source hippo/bin/activate
    pip install torch --index-url https://download.pytorch.org/whl/cu126
    pip install hipporag==2.0.0a4 --no-deps
    pip install -r requirements_hipporag.txt
    set OLLAMA_CONTEXT_LENGTH=16384 & ollama serve      # PowerShell: $env:OLLAMA_CONTEXT_LENGTH=16384

RUN
    python run_hipporag.py --datasets musique --n 3          # quick test first
    python run_hipporag.py                                   # 3 x 100 questions
    python run_hipporag.py --llm gemma4:latest --embedding nvidia/NV-Embed-v2

Resumable (Ctrl+C and relaunch). Results in results_hipporag/:
    perquery_<dataset>.jsonl   ranking, metrics and times for every question
    summary.csv                metrics and mean times per dataset
Send back the results_hipporag/ folder.

If HippoRAG rejects an argument, the script prints the real signature of its
constructor: send it back and the script will be adapted.
"""
import argparse
import csv
import inspect
import json
import os
import shutil
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ("hotpotqa", "2wiki", "musique")


# ═══════════════════ metrics (same as TestExperiments.py) ═══════════════════
def metrics(ranked, gold, n):
    seen = set(ranked)
    order = list(ranked) + [i for i in range(n) if i not in seen]   # missing -> appended
    rel = [i in gold for i in order]
    n_rel = sum(rel)
    out = {}
    for k in (3, 5):
        hit = sum(rel[:k])
        out[f"Recall@{k}"] = min(hit, n_rel) / n_rel if n_rel else 0.0
        out[f"FullRec@{k}"] = (1.0 if hit >= n_rel else 0.0) if n_rel else 0.0
    return out


# ═══════════════════ HippoRAG wrapper ═══════════════════
class MockHippo:
    """Fake HippoRAG, only to test this script without installing it."""
    def __init__(self, **kw):
        self.docs = []

    def index(self, docs):
        time.sleep(0.01)
        self.docs = list(docs)

    def retrieve(self, queries, num_to_retrieve):
        class Sol:
            pass
        s = Sol()
        s.docs = list(reversed(self.docs))[:num_to_retrieve]
        s.doc_scores = list(range(len(s.docs), 0, -1))
        return [s]


def stub_unused_modules():
    """HippoRAG imports vllm, gritlm, litellm and boto3 at load time even when they
    are not used. With the LLM served by Ollama and NV-Embed-v2 as embedding model
    none of them is needed, so on Windows (where vllm cannot be installed) they are
    replaced by empty modules. Real installed packages are left untouched."""
    import importlib
    import importlib.machinery
    import types
    def unavailable(*a, **k):
        raise RuntimeError("this optional package is not installed; it is not needed with Ollama")

    for name, attrs in (("vllm", ("SamplingParams", "LLM")), ("gritlm", ("GritLM",)),
                        ("litellm", ())):
        try:
            importlib.import_module(name)
        except ImportError:
            mod = types.ModuleType(name)
            # a real spec, so that importlib.util.find_spec (used by transformers) works
            mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
            for attr in attrs:
                base = Exception if attr.endswith("Error") else object
                setattr(mod, attr, type(attr, (base,), {} if base is Exception else {"__init__": unavailable}))
            sys.modules[name] = mod
            if "." in name:
                parent, child = name.rsplit(".", 1)
                setattr(sys.modules[parent], child, mod)
            print(f"[info] '{name}' not installed: replaced by an empty module (not used)")


_EMBEDDERS = {}


def _share_embedding_model():
    """HippoRAG loads the embedding model (NV-Embed-v2: 7B parameters) in every
    instance; we create one instance per question, so the model is loaded once
    and shared. Indexing itself is unchanged."""
    mod = sys.modules["hipporag.HippoRAG"]
    if getattr(mod, "_searchplus_shared", False):
        return
    original = mod._get_embedding_model_class

    def cached(embedding_model_name="nvidia/NV-Embed-v2"):
        cls = original(embedding_model_name=embedding_model_name)

        def factory(**kw):
            if embedding_model_name not in _EMBEDDERS:
                _EMBEDDERS[embedding_model_name] = cls(**kw)
            return _EMBEDDERS[embedding_model_name]
        return factory

    mod._get_embedding_model_class = cached
    mod._searchplus_shared = True


def make_hippo(save_dir, a):
    if a.mock:
        return MockHippo()
    stub_unused_modules()
    from hipporag import HippoRAG
    from hipporag.utils.config_utils import BaseConfig
    _share_embedding_model()
    cfg = BaseConfig()
    cfg.max_new_tokens = a.max_new_tokens   # thinking models need room (default 2048)
    kwargs = dict(global_config=cfg, save_dir=save_dir, llm_model_name=a.llm,
                  llm_base_url=a.llm_base_url, embedding_model_name=a.embedding)
    try:
        return HippoRAG(**kwargs)
    except TypeError:
        print("\nHippoRAG rejected the arguments. Its constructor is:")
        print("   HippoRAG" + str(inspect.signature(HippoRAG.__init__)))
        print("Send this line back so the script can be adapted.")
        raise


def run_question(item, a):
    paras = item["paragraphs"]
    docs = [f"{p['title']}\n{p['paragraph_text']}" for p in paras]
    gold = {i for i, p in enumerate(paras) if p["is_supporting"]}
    save = os.path.join(a.workdir, item["id"].replace("/", "_"))
    shutil.rmtree(save, ignore_errors=True)
    rec = {"id": item["id"], "n_passages": len(docs)}
    t0 = time.perf_counter()
    try:
        hr = make_hippo(save, a)
        hr.index(docs=docs)
        t1 = time.perf_counter()
        sol = hr.retrieve(queries=[item["question"]], num_to_retrieve=len(docs))
        sol = sol[0] if isinstance(sol, (list, tuple)) else sol
        t2 = time.perf_counter()
        returned = list(getattr(sol, "docs", []) or [])
        pos = {d.strip(): i for i, d in enumerate(docs)}
        ranked, unmatched = [], 0
        for d in returned:
            i = pos.get(str(d).strip())
            if i is None:
                unmatched += 1
            elif i not in ranked:
                ranked.append(i)
        rec.update({"ranking": ranked, "index_s": t1 - t0, "retrieve_s": t2 - t1,
                    "total_s": t2 - t0, "unmatched_docs": unmatched, "error": None})
    except Exception as e:  # counted as a failure: passages kept in their original order
        if isinstance(e, TypeError) and not a.mock and "unexpected keyword" in str(e):
            raise
        rec.update({"ranking": [], "index_s": None, "retrieve_s": None,
                    "total_s": time.perf_counter() - t0, "unmatched_docs": 0,
                    "error": f"{type(e).__name__}: {str(e)[:300]}"})
    rec["metrics"] = metrics(rec["ranking"], gold, len(docs))
    if not a.keep:
        shutil.rmtree(save, ignore_errors=True)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--n", type=int, default=100, help="questions per dataset (max 100)")
    ap.add_argument("--llm", default="gemma4:latest", help="Ollama tag, same model as Search+")
    ap.add_argument("--llm-base-url", default="http://localhost:11434/v1")
    ap.add_argument("--embedding", default="nvidia/NV-Embed-v2")
    ap.add_argument("--workdir", default=os.path.join(HERE, "hipporag_tmp"))
    ap.add_argument("--out", default="results_hipporag")
    ap.add_argument("--keep", action="store_true", help="keep HippoRAG's index folders")
    ap.add_argument("--max-new-tokens", type=int, default=4096,
                    help="LLM generation limit (HippoRAG default 2048; thinking tokens count too)")
    ap.add_argument("--mock", action="store_true", help="fake HippoRAG, to test the script")
    a = ap.parse_args()
    # HippoRAG puts the model name in folder names, and Windows forbids ':' there.
    # 'gemma4:latest' and 'gemma4' are the same model for Ollama.
    if a.llm.endswith(":latest"):
        a.llm = a.llm[:-len(":latest")]
    if ":" in a.llm and os.name == "nt":
        alias = a.llm.replace(":", "-")
        raise SystemExit(f"Windows does not allow ':' in folder names. Create an alias once with\n"
                         f"    ollama cp {a.llm} {alias}\n"
                         f"and run again with  --llm {alias}")

    os.environ.setdefault("OPENAI_API_KEY", "ollama")   # any non-empty key for a local server
    out = os.path.join(HERE, a.out)
    os.makedirs(out, exist_ok=True)
    data = json.load(open(os.path.join(HERE, "subsample.json"), encoding="utf-8"))["datasets"]
    print(f"llm={a.llm} @ {a.llm_base_url}   embedding={a.embedding}")
    if not a.mock:  # load the embedding model once, outside the timed region
        t = time.perf_counter()
        warm = os.path.join(a.workdir, "_warmup")
        make_hippo(warm, a)
        shutil.rmtree(warm, ignore_errors=True)
        print(f"embedding model loaded in {time.perf_counter() - t:.0f}s")

    t_start = time.time()
    for ds in a.datasets:
        items = data[ds][:a.n]
        path = os.path.join(out, f"perquery_{ds}.jsonl")
        done = set()
        if os.path.exists(path):  # keep successful questions; failed ones are run again
            recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
            ok = [r for r in recs if not r.get("error")]
            if len(ok) < len(recs):
                print(f"  {ds}: {len(recs) - len(ok)} failed questions will be run again")
                with open(path, "w", encoding="utf-8") as f:
                    f.writelines(json.dumps(r) + "\n" for r in ok)
            done = {r["id"] for r in ok}
        with open(path, "a", encoding="utf-8") as f:
            for k, item in enumerate(items):
                if item["id"] in done:
                    continue
                rec = run_question(item, a)
                f.write(json.dumps(rec) + "\n")
                f.flush()
                status = f"ERROR {rec['error']}" if rec["error"] else \
                    f"FR@5={rec['metrics']['FullRec@5']:.0f}  index {rec['index_s']:.1f}s"
                print(f"  {ds:9s} {k + 1:3d}/{len(items)}  {status}   "
                      f"elapsed {(time.time() - t_start) / 60:.1f} min", flush=True)

    rows = []
    for ds in a.datasets:
        path = os.path.join(out, f"perquery_{ds}.jsonl")
        if not os.path.exists(path):
            continue
        recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        ok = [r for r in recs if not r["error"]]
        row = {"dataset": ds, "queries": len(recs), "llm": a.llm, "embedding": a.embedding,
               "error_rate": round(1 - len(ok) / len(recs), 4) if recs else ""}
        for m in ("Recall@3", "Recall@5", "FullRec@3", "FullRec@5"):
            row[m] = round(statistics.mean(r["metrics"][m] for r in recs), 4)   # failures included
        for key in ("index_s", "retrieve_s", "total_s"):
            vals = [r[key] for r in ok if r.get(key) is not None]
            row[f"{key}_mean"] = round(statistics.mean(vals), 3) if vals else ""
        row["unmatched_docs"] = sum(r.get("unmatched_docs", 0) for r in recs)
        rows.append(row)
    if rows:
        with open(os.path.join(out, "summary.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print()
        for r in rows:
            print(f"{r['dataset']:9s} R@5 {r['Recall@5']}  FR@5 {r['FullRec@5']}  "
                  f"index {r['index_s_mean']} s/question  errors {r['error_rate']}")
        if any(r["unmatched_docs"] for r in rows):
            print("[warning] some returned passages could not be matched to the input: tell me")
        print(f"-> {a.out}/summary.csv")


if __name__ == "__main__":
    main()
