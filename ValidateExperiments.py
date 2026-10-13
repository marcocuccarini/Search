import json, os, sys, math, random, glob, time, itertools
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
import torch
from sentence_transformers import util

# Import EVERYTHING from config
from config import *

_MODEL_CACHE = {}
_EMB_CACHE = {}# Good practice to have this here too

# ── Utilità metriche ──────────────────────────────────────────────
def recall_at_k(ranked: List[bool], k: int) -> float:
    n_rel = sum(ranked)
    if n_rel == 0: return 0.0
    return min(sum(ranked[:k]), n_rel) / n_rel

def full_recall_at_k(ranked: List[bool], k: int) -> float:
    n_rel = sum(ranked)
    if n_rel == 0: return 0.0
    return 1.0 if sum(ranked[:k]) >= n_rel else 0.0

def ndcg_at_k(ranked: List[bool], k: int) -> float:
    def dcg(r, k):
        return sum(rel / math.log2(i + 2) for i, rel in enumerate(r[:k]))
    ideal = sorted(ranked, reverse=True)
    d = dcg(ranked, k)
    d_ideal = dcg(ideal, k)
    return d / d_ideal if d_ideal > 0 else 0.0

def mrr(ranked: List[bool]) -> float:
    for i, r in enumerate(ranked):
        if r: return 1.0 / (i + 1)
    return 0.0

def avg_metrics(results: List[Dict]) -> Dict:
    keys = ["Recall@3","Recall@5","Recall@10","FullRec@3","FullRec@5","FullRec@10","NDCG@10","MRR"]
    out = {}
    for k in keys:
        vals = [r[k] for r in results if k in r]
        out[k] = round(sum(vals)/len(vals), 4) if vals else 0.0
    return out

# ── Retrieval Helpers ──────────────────────────────────────────────

def filter_edges_by_type(edges: List[Dict], mode: str) -> List[Dict]:
    if mode == "triple":
        return [e for e in edges if e.get("relation_type") == "kg_bridge"]
    elif mode == "entity":
        return [e for e in edges if e.get("relation_type") == "entity_overlap"]
    return edges

def get_model(method_name):
    global _MODEL_CACHE
    model_id = MODEL_CONFIGS[method_name].get("model")
    if method_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- Loading {method_name} ({model_id}) on {device} ---")
        _MODEL_CACHE[method_name] = SentenceTransformer(model_id, device=device)
    return _MODEL_CACHE[method_name]

def apply_prefixes(method, query, texts):
    cfg = MODEL_CONFIGS[method]
    q = cfg.get("query_prefix", "") + query
    docs = [cfg.get("doc_prefix", "") + t for t in texts]
    return q, docs

def eval_ranking(ranked_indices: List[int], supporting_set: set, n_total: int) -> Dict:
    ranked_bool = [i in supporting_set for i in ranked_indices]
    seen = set(ranked_indices)
    # Ensure all nodes are accounted for in the ranking
    for i in range(n_total):
        if i not in seen:
            ranked_bool.append(i in supporting_set)
    return {
        "Recall@3": recall_at_k(ranked_bool, 3),
        "Recall@5": recall_at_k(ranked_bool, 5),
        "Recall@10": recall_at_k(ranked_bool, 10),
        "FullRec@3": full_recall_at_k(ranked_bool, 3),
        "FullRec@5": full_recall_at_k(ranked_bool, 5),
        "FullRec@10": full_recall_at_k(ranked_bool, 10),
        "NDCG@10": ndcg_at_k(ranked_bool, 10),
        "MRR": mrr(ranked_bool),
    }

# ── Core Algorithms ────────────────────────────────────────────────

def compute_ge_embeddings(graph: Dict, base_scores: List[float], ge_alpha: float, ge_hops: int, graph_type: str) -> List[float]:
    nodes = graph["nodes"]
    edges = filter_edges_by_type(graph.get("edges", []), graph_type)
    n = len(nodes)
    from collections import defaultdict

    edge_counts = defaultdict(int)

    for e in edges:
        src = e["src"]
        dst = e["dst"]

        edge_counts[(src, dst)] += 1
        edge_counts[(dst, src)] += 1

    adj = defaultdict(list)

    for (src, dst), count in edge_counts.items():
        adj[src].append((dst, count))

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

def bfs_retrieve_with_base(graph: Dict, base_scores: List[float], top_k_bm25: int, max_depth: int, branch: int, alpha: float, graph_type: str) -> List[int]:
    nodes = graph["nodes"]
    edges = filter_edges_by_type(graph.get("edges", []), graph_type)
    n = len(nodes)
    from collections import defaultdict

    edge_counts = defaultdict(int)

    for e in edges:
        src = e["src"]
        dst = e["dst"]

        edge_counts[(src, dst)] += 1
        edge_counts[(dst, src)] += 1

    adj = defaultdict(list)

    for (src, dst), count in edge_counts.items():
        adj[src].append((dst, count))

    ranked_base = sorted(range(n), key=lambda i: base_scores[i], reverse=True)
    seeds = ranked_base[:top_k_bm25]
    visited, v_set, frontier = list(seeds), set(seeds), list(seeds)
    max_s = max(base_scores) if max(base_scores) > 0 else 1.0

    for _ in range(max_depth):
        if not frontier: break
        next_f = []
        for node in frontier:
            scored_neighs = []
            for j, w in adj[node]:
                if j in v_set: continue
                score = (alpha * (w/n)) + ((1-alpha) * (base_scores[j]/max_s))
                scored_neighs.append((j, score))
            
            scored_neighs.sort(key=lambda x: x[1], reverse=True)
            for j, _ in scored_neighs[:branch]:
                if j not in v_set:
                    v_set.add(j); visited.append(j); next_f.append(j)
        frontier = next_f
    
    for i in ranked_base:
        if i not in v_set: visited.append(i)
    return visited

# ── Main Execution ────────────────────────────────────────────────

from tqdm import tqdm


def run_dataset_validation(dataset_name: str, data_path: str, graphs_path: str):
    print(f"\n{'='*30} STARTING: {dataset_name} {'='*30}")
    with open(graphs_path, encoding="utf-8") as f: all_graphs = json.load(f)
    items = []
    with open(data_path, encoding="utf-8") as f:
        for line in f: items.append(json.loads(line))
    
    if SAMPLE_SIZE: items = items[:SAMPLE_SIZE]
    eval_items = [it for it in items if it["id"] in all_graphs]
    
    all_results = []
    methods_to_test = [m for m, enabled in ENABLED_METHODS.items() if enabled]

    for method in methods_to_test:
        print(f"\n--- Method: {method} ---")
        item_base_scores = {}
        cfg = MODEL_CONFIGS[method]
# 1. PRE-COMPUTE BASE SCORES (Efficient Global Batching)
        if cfg["type"] == "dense":
            model = get_model(method)
            
            # Collect unique texts and questions to avoid redundant work
            all_queries = [it["question"] for it in eval_items]
            unique_docs = set()
            for it in eval_items:
                g = all_graphs[it["id"]]
                for nd in g["nodes"]:
                    unique_docs.add(nd["title"] + " " + nd["text"])
            
            unique_docs = list(unique_docs)
            doc_to_idx = {txt: i for i, txt in enumerate(unique_docs)}

            # Apply prefixes (once for everything)
            q_p_list = [cfg.get("query_prefix", "") + q for q in all_queries]
            d_p_list = [cfg.get("doc_prefix", "") + d for d in unique_docs]

            # Massive GPU Batching
            print(f"--- Encoding {len(q_p_list)} queries and {len(d_p_list)} unique docs ---")
            q_embs = model.encode(q_p_list, convert_to_tensor=True, show_progress_bar=True, batch_size=128)
            d_embs = model.encode(d_p_list, convert_to_tensor=True, show_progress_bar=True, batch_size=128)

            # Map back to items
            for idx, it in enumerate(eval_items):
                g = all_graphs[it["id"]]
                # Get the indices of the docs belonging to this specific graph
                doc_indices = [doc_to_idx[nd["title"] + " " + nd["text"]] for nd in g["nodes"]]
                
                # Extract pre-computed embeddings and calc similarity
                current_q_emb = q_embs[idx:idx+1]
                current_d_embs = d_embs[doc_indices]
                
                item_base_scores[it["id"]] = util.cos_sim(current_q_emb, current_d_embs)[0].cpu().tolist()

        else:
            # BM25 Efficiency: Collect all nodes once if the graph corpus is shared
            # If graphs are strictly local to questions, you can still wrap the whole loop
            from rank_bm25 import BM25Okapi
            for it in tqdm(eval_items, desc="BM25 Scoring"):
                g = all_graphs[it["id"]]
                tokenized_corpus = [nd["title"].lower().split() + nd["text"].lower().split() for nd in g["nodes"]]
                bm25 = BM25Okapi(tokenized_corpus)
                item_base_scores[it["id"]] = bm25.get_scores(it["question"].lower().split()).tolist()

        # 2. GRID SEARCH
        for mode in ["none", "bfs", "ge"]:
            if mode == "none":
                res = []
                for it in eval_items:
                    scores = item_base_scores[it["id"]]
                    supp = {i for i, nd in enumerate(all_graphs[it["id"]]["nodes"]) if nd["is_supporting"]}
                    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                    res.append(eval_ranking(ranked, supp, len(scores)))
                avg = avg_metrics(res)
                all_results.append({"method": method, "graph_mode": "none", "metrics": avg})
                print(f" [BASE] {PRIMARY_METRIC}: {avg[PRIMARY_METRIC]}")

            elif mode == "bfs":
                print(f" Running BFS Grid Search for {method}...")
                for g_type in GRID_GRAPH_TYPE:
                    for b_cfg in BFS_CONFIGS:
                        res = []
                        for it in eval_items:
                            ranked = bfs_retrieve_with_base(all_graphs[it["id"]], item_base_scores[it["id"]], **b_cfg, graph_type=g_type)
                            supp = {i for i, nd in enumerate(all_graphs[it["id"]]["nodes"]) if nd["is_supporting"]}
                            res.append(eval_ranking(ranked, supp, len(item_base_scores[it["id"]])))
                        avg = avg_metrics(res)
                        all_results.append({"method": method, "graph_mode": "bfs", "graph_type": g_type, "config": b_cfg, "metrics": avg})
                        
                        # NEW: Print BFS result to terminal
                        print(f"  [BFS] Type: {g_type:6} | k:{b_cfg['top_k_bm25']} d:{b_cfg['max_depth']} b:{b_cfg['branch']} a:{b_cfg['alpha']} -> {PRIMARY_METRIC}: {avg[PRIMARY_METRIC]}")

            elif mode == "ge":
                print(f" Running GE Grid Search for {method}...")
                for g_type in GRID_GRAPH_TYPE:
                    for g_cfg in GE_CONFIGS:
                        res = []
                        for it in eval_items:
                            scores = compute_ge_embeddings(all_graphs[it["id"]], item_base_scores[it["id"]], **g_cfg, graph_type=g_type)
                            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                            supp = {i for i, nd in enumerate(all_graphs[it["id"]]["nodes"]) if nd["is_supporting"]}
                            res.append(eval_ranking(ranked, supp, len(scores)))
                        avg = avg_metrics(res)
                        all_results.append({"method": method, "graph_mode": "ge", "graph_type": g_type, "config": g_cfg, "metrics": avg})
                        
                        # NEW: Print GE result to terminal
                        print(f"  [GE ] Type: {g_type:6} | alpha:{g_cfg['ge_alpha']} hops:{g_cfg['ge_hops']} -> {PRIMARY_METRIC}: {avg[PRIMARY_METRIC]}")
    # Summary and Save
    best_res = max(all_results, key=lambda x: x["metrics"][PRIMARY_METRIC])
    print(f"\nWINNER: {best_res['method']} via {best_res['graph_mode']} ({best_res.get('graph_type', 'N/A')})")
    print(f"Result: {best_res['metrics'][PRIMARY_METRIC]}")

    out_path = os.path.join(RESULTS_DIR, f"results_{dataset_name}.json")
    with open(out_path, "w") as f: json.dump(all_results, f, indent=2)

def main():
    all_data = glob.glob(os.path.join(DATA_DIR, "*.jsonl"))
    for d_path in all_data:
        stem = os.path.splitext(os.path.basename(d_path))[0]
        g_path = os.path.join(GRAPHS_DIR, stem + "_hybrid_graphs.json")
        if os.path.exists(g_path):
            run_dataset_validation(stem, d_path, g_path)

if __name__ == "__main__":
    main()