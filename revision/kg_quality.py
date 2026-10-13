"""KG quality evaluation and its link to the reranking gain (reviewer point 6).

No gold KG exists, so quality is measured against what the datasets do provide
(gold supporting passages, their titles, and the question decompositions):

Intrinsic (KG only)
  parse_ok            the LLM output was valid JSON (no "error" key)
  n_triples           number of extracted triples
  grounding           share of distinct triple endpoints that occur in some passage
                      (with the repo's own substring matcher); the rest can never
                      create an edge (hallucinated, paraphrased or badly formatted)
  hub_share           share of grounded entities that occur in >= half the passages
                      (generic entities such as 'film' or '1995' that connect everything)
Coverage of what the question needs
  gold_title_recall   share of gold passages whose title appears among KG entities
  bridge_recall       share of bridge entities (answers of non-final hops in the
                      MuSiQue / 2Wiki decompositions) that appear among KG entities
Graph vs. gold passages (uses the built graph, edge type 'both')
  gold_pair_recall    share of gold passage pairs that are directly connected
  gold_connected      all gold passages lie in one connected component
  edge_precision      share of connected pairs that join two gold passages
  distractor_weight   share of edge weight that touches at least one distractor
  gold_isolated       share of gold passages with no edge at all

Then, per query, the quality measures are related to the reranking gain
(FR@5 and R@5 of BFS / NSP minus the retriever baseline), using the per-query
results written by run_experiments.py.

    python revision/kg_quality.py --methods BM25 BGE-large
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from BuildGraphsPerQuestion import _entity_in_text, _get_triplet  # noqa: E402
from common import OUT, ensure_hashseed, write_csv, write_json  # noqa: E402
from core import DATASETS, load_items, load_kg  # noqa: E402
from graph_variants import get_graphs  # noqa: E402

KG_SETS = {"aware": "dataKG", "unaware": "dataKG_without"}
QUALITY = ["parse_ok", "n_triples", "grounding", "hub_share", "gold_title_recall",
           "bridge_recall", "gold_pair_recall", "gold_connected", "edge_precision",
           "distractor_weight", "gold_isolated"]


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[_\W]+", " ", str(s).lower())).strip()


def kg_entities(kg_item):
    if not kg_item:
        return [], [], False
    kg = kg_item.get("knowledge_graph", kg_item)
    ok = isinstance(kg, dict) and "error" not in kg
    triples = []
    for e in (kg.get("edges", []) if isinstance(kg, dict) else []) or []:
        t = _get_triplet(e)
        if t and t[0] and t[2]:
            triples.append(t)
    ents = sorted({t[0] for t in triples} | {t[2] for t in triples})
    return triples, ents, ok


def bridge_entities(item):
    """Answers of non-final hops that later hops build on."""
    dec = item.get("question_decomposition") or []
    out = []
    for k, step in enumerate(dec[:-1]):
        ans = str(step.get("answer", "")).strip()
        if not ans or ans.lower().startswith("check supporting"):
            continue
        later = " ".join(str(s.get("question", "")) for s in dec[k + 1:])
        if f"#{k + 1}" in later or norm(ans) and norm(ans) in norm(later):
            out.append(ans)
    return out


def matches(entity_list, target):
    t = norm(target)
    if len(t) < 3:
        return False
    for e in entity_list:
        n = norm(e)
        if len(n) >= 3 and (n == t or t in n or n in t):
            return True
    return False


def graph_measures(graph):
    n = len(graph["nodes"])
    gold = [i for i, nd in enumerate(graph["nodes"]) if nd["is_supporting"]]
    W = np.zeros((n, n))
    for e in graph.get("edges", []):
        W[e["src"], e["dst"]] += 1
        W[e["dst"], e["src"]] += 1
    A = W > 0
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j]]
    gold_set = set(gold)
    gp = [(i, j) for k, i in enumerate(gold) for j in gold[k + 1:]]
    # components
    comp, c = [-1] * n, 0
    for s in range(n):
        if comp[s] < 0:
            stack, comp[s] = [s], c
            while stack:
                u = stack.pop()
                for v in np.nonzero(A[u])[0]:
                    if comp[v] < 0:
                        comp[v] = c
                        stack.append(v)
            c += 1
    tot_w = W[np.triu_indices(n, 1)].sum()
    dis_w = sum(W[i, j] for i, j in pairs if i not in gold_set or j not in gold_set)
    return {
        "gold_pair_recall": float(np.mean([A[i, j] for i, j in gp])) if gp else float("nan"),
        "gold_connected": float(len({comp[g] for g in gold}) <= 1) if gold else float("nan"),
        "edge_precision": float(np.mean([i in gold_set and j in gold_set for i, j in pairs]))
        if pairs else float("nan"),
        "distractor_weight": float(dis_w / tot_w) if tot_w else float("nan"),
        "gold_isolated": float(np.mean([not A[g].any() for g in gold])) if gold else float("nan"),
        "n_connected_pairs": len(pairs),
    }


def quality_per_query(items, kg_map, graphs):
    out = []
    for it in items:
        triples, ents, ok = kg_entities(kg_map.get(it["id"]))
        paras = it["paragraphs"]
        found = {e: [p["idx"] for p in paras if _entity_in_text(e, p["paragraph_text"])
                     or _entity_in_text(e, p["title"])] for e in ents}
        grounded = [e for e in ents if found[e]]
        gold_titles = [p["title"] for p in paras if p["is_supporting"]]
        bridges = bridge_entities(it)
        r = {"qid": it["id"], "parse_ok": float(ok), "n_triples": len(triples),
             "grounding": len(grounded) / len(ents) if ents else float("nan"),
             "hub_share": (float(np.mean([len(found[e]) >= len(paras) / 2 for e in grounded]))
                           if grounded else float("nan")),
             "gold_title_recall": float(np.mean([matches(ents, t) for t in gold_titles]))
             if gold_titles else float("nan"),
             "bridge_recall": float(np.mean([matches(ents, b) for b in bridges]))
             if bridges else float("nan"), "n_bridges": len(bridges)}
        r.update(graph_measures(graphs[it["id"]]))
        out.append(r)
    return out


def nanmean(x):
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan")
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ensure_hashseed()
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["BM25"])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    a = ap.parse_args()
    summary, link_rows, group_rows = [], [], []
    for ds in a.datasets:
        items = load_items(ds, "test")
        per_set = {}
        for name, kg_dir in KG_SETS.items():
            graphs = get_graphs(name, ds, "test", items)
            q = quality_per_query(items, load_kg(kg_dir, ds, "test"), graphs)
            per_set[name] = q
            write_csv(os.path.join(OUT, "kg_quality", f"perquery_{ds}_{name}.csv"), q)
            row = {"dataset": ds, "kg": name, "queries": len(q)}
            for m in QUALITY + ["n_connected_pairs"]:
                row[m] = round(nanmean([r[m] for r in q]), 4)
            summary.append(row)
            print(ds, name, {k: row[k] for k in ("parse_ok", "grounding", "bridge_recall",
                                                "gold_pair_recall", "edge_precision")}, flush=True)

        # ---- link quality (aware KG) to the per-query gain
        q = per_set["aware"]
        for method in a.methods:
            path = os.path.join(OUT, "perquery", f"{ds}_{method}.json")
            if not os.path.exists(path):
                print(f"  (no {path}; run run_experiments.py first)")
                continue
            perq = json.load(open(path))
            idx = {qid: k for k, qid in enumerate(perq["qids"])}
            order = [idx[r["qid"]] for r in q]
            for mode in ("bfs", "ge"):
                key = f"aware|{mode}|both"
                for metric in ("FullRec@5", "Recall@5"):
                    gain = np.array(perq[key][metric])[order] - np.array(perq["none"][metric])[order]
                    lr = {"dataset": ds, "retriever": method, "mode": mode, "metric": metric,
                          "mean_gain": round(float(gain.mean()), 4),
                          "share_improved": round(float((gain > 0).mean()), 4),
                          "share_hurt": round(float((gain < 0).mean()), 4)}
                    for m in QUALITY:
                        lr[f"rho_{m}"] = round(spearman([r[m] for r in q], gain), 3)
                    link_rows.append(lr)
                    # gain split by the two most interpretable measures
                    for m, label in (("gold_connected", "gold passages connected"),
                                     ("bridge_recall", "bridge entity in KG")):
                        vals = np.array([r[m] for r in q], float)
                        for flag, sel in (("yes", vals >= 0.999), ("no", vals < 0.999)):
                            sel = sel & np.isfinite(vals)
                            if sel.sum() == 0:
                                continue
                            group_rows.append({"dataset": ds, "retriever": method, "mode": mode,
                                               "metric": metric, "split_by": label, "value": flag,
                                               "queries": int(sel.sum()),
                                               "mean_gain": round(float(gain[sel].mean()), 4),
                                               "share_hurt": round(float((gain[sel] < 0).mean()), 4)})
    write_csv(os.path.join(OUT, "kg_quality", "summary.csv"), summary)
    write_csv(os.path.join(OUT, "kg_quality", "quality_vs_gain_correlation.csv"), link_rows)
    write_csv(os.path.join(OUT, "kg_quality", "gain_by_quality_group.csv"), group_rows)


if __name__ == "__main__":
    main()
