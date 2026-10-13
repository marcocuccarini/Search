"""Find and illustrate queries where a poor KG makes Search+ rank WORSE than the
retriever alone (reviewer request: an example of a bad KG hurting reranking).

For each test query it re-runs BFS (the tuned 'aware|bfs|both' config from
run_experiments.py) while recording which head expanded to which passage and
through which entities, then keeps queries where a gold passage leaves the
top-5 and a distractor enters it through the graph. Candidates are scored by a
'blame' diagnosis:

  substring   the edge exists only because of the repo's substring matcher
              (entity found inside another word, e.g. 'US' in 'focuses')
  generic     the entity occurs in >= half of the passages (hub)
  ungrounded  the gold passage has no edge because its entities were not
              extracted / not found (missing relation)

Writes revision_results/bad_examples/<ds>_<retriever>.json with all candidates,
and for the best one: a figure (PNG + PDF), a ranking table and the passages in
LaTeX, ready for the paper.

    python revision/bad_example.py --dataset musique --method BGE-large
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, ensure_hashseed  # noqa: E402
from core import eval_ranking, filter_edges_by_type, load_items, supporting  # noqa: E402
from graph_variants import get_graphs  # noqa: E402
from scores import get_scores  # noqa: E402


def edge_entity(e):
    return e.get("shared_entity") or e.get("subject") or ""


def word_match(entity, text):
    return re.search(r"(?<!\w)" + re.escape(str(entity).lower()) + r"(?!\w)", str(text).lower()) is not None


def traced_bfs(graph, base_scores, top_k_bm25, max_depth, branch, alpha, graph_type):
    """Identical to bfs_retrieve_with_base, plus a trace of why each node was added."""
    nodes = graph["nodes"]
    edges = filter_edges_by_type(graph.get("edges", []), graph_type)
    n = len(nodes)
    adj_strength = defaultdict(lambda: defaultdict(float))
    why = defaultdict(list)
    for e in edges:
        src, dst = e["src"], e["dst"]
        adj_strength[src][dst] += 1.0
        adj_strength[dst][src] += 1.0
        why[(min(src, dst), max(src, dst))].append(e)
    adj = defaultdict(list)
    for src in adj_strength:
        for dst, strength in adj_strength[src].items():
            adj[src].append((dst, strength))
    ranked_base = sorted(range(n), key=lambda i: base_scores[i], reverse=True)
    seeds = ranked_base[:top_k_bm25]
    visited, v_set, frontier = list(seeds), set(seeds), list(seeds)
    trace = {s: {"via": None, "depth": 0} for s in seeds}
    max_s = max(base_scores) if max(base_scores) > 0 else 1.0
    for depth in range(max_depth):
        if not frontier:
            break
        next_f = []
        for node in frontier:
            scored = []
            for j, w in adj[node]:
                if j in v_set:
                    continue
                score = alpha * (w / n) + (1 - alpha) * (base_scores[j] / max_s)
                scored.append((j, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            for j, _ in scored[:branch]:
                if j not in v_set:
                    v_set.add(j)
                    visited.append(j)
                    next_f.append(j)
                    trace[j] = {"via": node, "depth": depth + 1,
                                "edges": why[(min(node, j), max(node, j))]}
        frontier = next_f
    for i in ranked_base:
        if i not in v_set:
            visited.append(i)
    return visited, trace, why


def diagnose(graph, item, trace, why, base_rank, new_rank):
    nodes, n = graph["nodes"], len(graph["nodes"])
    gold = supporting(graph)
    top_b, top_n = set(base_rank[:5]), set(new_rank[:5])
    lost = sorted(g for g in gold if g in top_b and g not in top_n)
    intruders = sorted(i for i in top_n - top_b if i not in gold)
    reasons = []
    for i in intruders:
        t = trace.get(i, {})
        for e in t.get("edges", []):
            inn = lambda ent, nd: word_match(ent, nd["text"]) or word_match(ent, nd["title"])  # noqa: E731
            if e["relation_type"] == "kg_bridge":
                ends = [(e.get("subject", ""), nodes[e["src"]]), (e.get("object", ""), nodes[e["dst"]])]
                ent = f"{e.get('subject')} -[{e.get('relation')}]-> {e.get('object')}"
            else:
                ends = [(e.get("shared_entity", ""), nodes[e["src"]]),
                        (e.get("shared_entity", ""), nodes[e["dst"]])]
                ent = e.get("shared_entity", "")
            whole = all(inn(x, nd) for x, nd in ends)
            freq = max(sum(1 for nd in nodes if x and (x.lower() in nd["text"].lower()
                                                       or x.lower() in nd["title"].lower()))
                       for x, _ in ends)
            if not whole:
                kind = "substring"
            elif freq >= n / 2:
                kind = "generic"
            elif e["relation_type"] == "kg_bridge":
                kind = "off_query_relation"
            else:
                kind = "off_query_entity"
            reasons.append({"intruder": i, "via": t["via"], "edge": ent,
                            "relation_type": e["relation_type"], "kind": kind,
                            "entity_in_n_passages": freq})
    for g in lost:
        deg = sum(1 for (a, b) in why if g in (a, b))
        if deg == 0:
            reasons.append({"lost_gold": g, "kind": "ungrounded",
                            "note": "gold passage has no edge (its entities were not extracted or not matched)"})
    return lost, intruders, reasons


def draw(graph, base_rank, new_rank, trace, why, path_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    nodes = graph["nodes"]
    gold = supporting(graph)
    G = nx.Graph()
    used = set(base_rank[:7]) | set(new_rank[:7])
    for i in used:
        G.add_node(i)
    for (a, b), es in why.items():
        if a in used and b in used:
            G.add_edge(a, b, label=", ".join(sorted({edge_entity(e) for e in es}))[:28], w=len(es))
    pos = nx.spring_layout(G, seed=3, k=1.2)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    expansion = {(min(t["via"], j), max(t["via"], j)) for j, t in trace.items() if t.get("via") is not None}
    ecolor = ["#c0392b" if (min(u, v), max(u, v)) in expansion else "#bbbbbb" for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=ecolor,
                           width=[1 + 0.4 * G[u][v]["w"] for u, v in G.edges()])
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=700,
                           node_color=["#f4d03f" if i in gold else "#d6eaf8" for i in G.nodes()],
                           edgecolors=["#000" if i in new_rank[:5] else "#888" for i in G.nodes()],
                           linewidths=[2.2 if i in new_rank[:5] else 0.8 for i in G.nodes()])
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, ax=ax, font_size=7,
                                 edge_labels={(u, v): d["label"] for u, v, d in G.edges(data=True)})
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path_png, dpi=200)
    fig.savefig(path_png.replace(".png", ".pdf"))
    plt.close(fig)


def tex_escape(s):
    return (str(s).replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%")
            .replace("$", "\\$").replace("#", "\\#").replace("_", "\\_").replace("{", "\\{")
            .replace("}", "\\}"))


def main():
    ensure_hashseed()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="musique")
    ap.add_argument("--method", default="BM25")
    ap.add_argument("--pick", type=int, default=0, help="which candidate to illustrate (0 = best)")
    a = ap.parse_args()
    ds, method = a.dataset, a.method
    items = load_items(ds, "test")
    graphs = get_graphs("aware", ds, "test", items)
    items = [it for it in items if it["id"] in graphs]
    scores = get_scores(method, ds, "test", items, graphs)
    perq_path = os.path.join(OUT, "perquery", f"{ds}_{method}.json")
    if os.path.exists(perq_path):
        # config tuned on validation by run_experiments.py
        cfg = json.load(open(perq_path))["aware|bfs|both"]["config"]
        print(f"BFS config from run_experiments.py: {cfg}")
    else:
        # no run_experiments.py output: use the configuration reported in the paper
        from common import paper_cfg
        cfg = paper_cfg(ds, method, "bfs", "both")
        if cfg is None:
            raise SystemExit(f"No BFS config for {method} on {ds}: run "
                             f"run_experiments.py --methods {method} first")
        print(f"BFS config from the paper's appendix table: {cfg}")
    by_id = {it["id"]: it for it in items}

    cands = []
    for it in items:
        g, s = graphs[it["id"]], scores[it["id"]]
        base = sorted(range(len(s)), key=lambda i: s[i], reverse=True)
        new, trace, why = traced_bfs(g, s, **cfg, graph_type="both")
        mb, mn = eval_ranking(base, supporting(g), len(s)), eval_ranking(new, supporting(g), len(s))
        if mn["Recall@5"] >= mb["Recall@5"]:
            continue
        lost, intruders, reasons = diagnose(g, it, trace, why, base, new)
        if not lost:
            continue
        kinds = {r["kind"] for r in reasons}
        score = (mb["Recall@5"] - mn["Recall@5"]) + 0.5 * (mb["FullRec@5"] - mn["FullRec@5"]) \
            + 0.3 * ("substring" in kinds) + 0.2 * ("generic" in kinds) + 0.1 * ("ungrounded" in kinds) \
            - 0.01 * len(why)
        cands.append({"qid": it["id"], "question": it["question"], "score": round(score, 3),
                      "R@5_base": mb["Recall@5"], "R@5_bfs": mn["Recall@5"],
                      "FR@5_base": mb["FullRec@5"], "FR@5_bfs": mn["FullRec@5"],
                      "base_top5": base[:5], "bfs_top5": new[:5], "gold": sorted(supporting(g)),
                      "lost_gold": lost, "intruders": intruders, "reasons": reasons,
                      "n_connected_pairs": len(why)})
    cands.sort(key=lambda c: -c["score"])
    out_dir = os.path.join(OUT, "bad_examples")
    os.makedirs(out_dir, exist_ok=True)
    kinds = defaultdict(int)
    for c in cands:
        for k in {r["kind"] for r in c["reasons"]}:
            kinds[k] += 1
    summary = {"dataset": ds, "retriever": method, "bfs_config": cfg, "test_queries": len(items),
               "queries_hurt_R@5": len(cands), "failure_kinds": dict(kinds), "candidates": cands}
    with open(os.path.join(out_dir, f"{ds}_{method}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"{len(cands)}/{len(items)} queries where BFS lowers R@5 and a gold passage leaves the top 5")
    print("failure kinds (queries):", dict(kinds))
    if not cands:
        return

    c = cands[a.pick]
    it, g, s = by_id[c["qid"]], graphs[c["qid"]], scores[c["qid"]]
    base = sorted(range(len(s)), key=lambda i: s[i], reverse=True)
    new, trace, why = traced_bfs(g, s, **cfg, graph_type="both")
    stem = os.path.join(out_dir, f"{ds}_{method}_{c['qid']}")
    draw(g, base, new, trace, why, stem + ".png",
         f"{ds} {c['qid']} (gold = yellow, bold = BFS top-5, red = BFS expansion)")
    gold = supporting(g)
    star = lambda i: f"{i}$^\\star$" if i in gold else str(i)  # noqa: E731
    used = sorted(set(base[:7]) | set(new[:7]))
    lines = ["% Bad-KG example generated by revision/bad_example.py",
             f"% Query {c['qid']} ({ds}, retriever {method}, BFS config {cfg})",
             f"% Question: {it['question']}", "",
             "\\begin{table}[t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{lccccc}", "\\toprule", "Method & 1 & 2 & 3 & 4 & 5 \\\\", "\\midrule",
             "Base & " + " & ".join(star(i) for i in base[:5]) + " \\\\",
             "BFS & " + " & ".join(star(i) for i in new[:5]) + " \\\\",
             "\\bottomrule", "\\end{tabular}",
             "\\caption{Ranking sequences for a query where a low-quality KG hurts BFS. "
             "$^\\star$ = relevant.}", f"\\label{{tab:bad_example}}", "\\end{table}", "",
             "% Why each intruder entered the top 5:"]
    for r in c["reasons"]:
        lines.append("%   " + json.dumps(r))
    lines += ["", "\\begin{table*}[t]", "\\centering", "\\begin{tabularx}{\\textwidth}{c X}",
              "\\toprule", "Node idx & Text \\\\", "\\midrule"]
    for i in used:
        nd = g["nodes"][i]
        lines.append(f"{star(i)} & \\textbf{{{tex_escape(nd['title'])}}}: {tex_escape(nd['text'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabularx}", "\\caption{Passages of the bad-KG example.}",
              "\\label{tab:bad_example_text}", "\\end{table*}"]
    with open(stem + ".tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    kg = json.load(open(os.path.join(os.path.dirname(OUT), "dataKG", f"{ds}_validation_test_KG.json")))
    with open(stem + "_kg.json", "w") as f:
        json.dump(kg.get(c["qid"]), f, indent=1)
    print(json.dumps({k: c[k] for k in ("qid", "question", "base_top5", "bfs_top5", "gold",
                                        "lost_gold", "intruders")}, indent=1))
    for r in c["reasons"]:
        print("  ", r)
    print(f"  -> {stem}.png/.pdf/.tex/_kg.json")


if __name__ == "__main__":
    main()
