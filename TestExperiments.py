"""
TestExperiments.py
==================
Reads the best hyperparameter configs found by ValidateExperiments.py
(from results/validation_best_combined_*_val.json) and evaluates each
combo on the held-out test split.

Always recomputes every combo — no skip logic, no cache.

Output per combo × dataset  : results/test_predictions_{method}_{mode}_{graph_type}_{dataset}.json
Output master summary        : results/test_results_by_combo.json  (merged, never overwritten)

Run order
---------
1. ValidateExperiments.py   → produces validation_best_combined_{stem}.json
2. TestExperiments.py       → reads those files, produces test_predictions_*.json
"""

import json
import os
import math
from collections import defaultdict
from typing import List, Dict

import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from rank_bm25 import BM25Okapi

from config import (
    MODEL_CONFIGS, GRID_GRAPH_TYPE,
    GRID_BFS, GRID_GE, BFS_CONFIGS, GE_CONFIGS,
    DATA_DIR, GRAPHS_DIR, RESULTS_DIR,
    SAMPLE_SIZE, PRIMARY_METRIC,
)

_MODEL_CACHE: Dict = {}


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def recall_at_k(ranked: List[bool], k: int) -> float:
    n_rel = sum(ranked)
    if n_rel == 0:
        return 0.0
    return min(sum(ranked[:k]), n_rel) / n_rel


def full_recall_at_k(ranked: List[bool], k: int) -> float:
    n_rel = sum(ranked)
    if n_rel == 0:
        return 0.0
    return 1.0 if sum(ranked[:k]) >= n_rel else 0.0


def ndcg_at_k(ranked: List[bool], k: int) -> float:
    def dcg(r, kk):
        return sum(rel / math.log2(i + 2) for i, rel in enumerate(r[:kk]))
    ideal = sorted(ranked, reverse=True)
    d = dcg(ranked, k)
    d_ideal = dcg(ideal, k)
    return d / d_ideal if d_ideal > 0 else 0.0


def mrr(ranked: List[bool]) -> float:
    for i, r in enumerate(ranked):
        if r:
            return 1.0 / (i + 1)
    return 0.0


def avg_metrics(results: List[Dict]) -> Dict:
    keys = ["Recall@3", "Recall@5", "Recall@10",
            "FullRec@3", "FullRec@5", "FullRec@10",
            "NDCG@10", "MRR"]
    return {
        k: round(sum(r[k] for r in results) / len(results), 4)
        if results else 0.0
        for k in keys
    }


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def filter_edges_by_type(edges: List[Dict], mode: str) -> List[Dict]:
    if mode == "triple":
        return [e for e in edges if e.get("relation_type") == "kg_bridge"]
    elif mode == "entity":
        return [e for e in edges if e.get("relation_type") == "entity_overlap"]
    return edges  # "both" / "all"


def get_model(method_name: str) -> SentenceTransformer:
    if method_name not in _MODEL_CACHE:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = MODEL_CONFIGS[method_name]["model"]
        print(f"--- Loading {method_name} ({model_id}) on {device} ---")
        _MODEL_CACHE[method_name] = SentenceTransformer(model_id, device=device)
    return _MODEL_CACHE[method_name]


def eval_ranking(ranked: List[int], supp: set, n: int) -> Dict:
    ranked_bool = [i in supp for i in ranked]
    seen = set(ranked)
    for i in range(n):
        if i not in seen:
            ranked_bool.append(i in supp)
    return {
        "Recall@3":   recall_at_k(ranked_bool, 3),
        "Recall@5":   recall_at_k(ranked_bool, 5),
        "Recall@10":  recall_at_k(ranked_bool, 10),
        "FullRec@3":  full_recall_at_k(ranked_bool, 3),
        "FullRec@5":  full_recall_at_k(ranked_bool, 5),
        "FullRec@10": full_recall_at_k(ranked_bool, 10),
        "NDCG@10":    ndcg_at_k(ranked_bool, 10),
        "MRR":        mrr(ranked_bool),
    }


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH ALGORITHMS
# ══════════════════════════════════════════════════════════════════════════════

def compute_ge_embeddings(graph: Dict, base_scores: List[float],
                          ge_alpha: float, ge_hops: int,
                          graph_type: str) -> List[float]:
    nodes = graph["nodes"]
    edges = filter_edges_by_type(graph.get("edges", []), graph_type)
    n = len(nodes)
    adj_strength = defaultdict(lambda: defaultdict(float))

    for e in edges:
        src, dst = e["src"], e["dst"]

        # each edge contributes +1
        adj_strength[src][dst] += 1.0
        adj_strength[dst][src] += 1.0

    adj = defaultdict(list)
    for src in adj_strength:
        for dst, strength in adj_strength[src].items():
            adj[src].append((dst, strength))

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


def bfs_retrieve_with_base(graph: Dict, base_scores: List[float],
                           top_k_bm25: int, max_depth: int,
                           branch: int, alpha: float,
                           graph_type: str) -> List[int]:
    nodes = graph["nodes"]
    edges = filter_edges_by_type(graph.get("edges", []), graph_type)
    n = len(nodes)
    adj_strength = defaultdict(lambda: defaultdict(float))

    for e in edges:
        src, dst = e["src"], e["dst"]

        # each edge contributes +1
        adj_strength[src][dst] += 1.0
        adj_strength[dst][src] += 1.0

    adj = defaultdict(list)
    for src in adj_strength:
        for dst, strength in adj_strength[src].items():
            adj[src].append((dst, strength))
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
                score = alpha * (w / n) + (1 - alpha) * (base_scores[j] / max_s)
                scored.append((j, score))
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


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION LOADER  (handles both list and dict output formats)
# ══════════════════════════════════════════════════════════════════════════════

def load_validation_results(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return list(data.values())
    raise ValueError(f"Unexpected format in {path}: {type(data)}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TEST RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_test_with_best_config(validation_results_file: str,
                              test_dataset_path: str,
                              test_graphs_path: str,
                              dataset_label: str = "dataset"):
    print("=" * 80)
    print(f"DATASET  : {dataset_label}")
    print(f"VAL FILE : {os.path.basename(validation_results_file)}")
    print(f"TEST DATA: {os.path.basename(test_dataset_path)}")
    print("=" * 80)

    # ── Load validation results & pick best config per combo ──────────────
    validation_results = load_validation_results(validation_results_file)
    best_by_combo: Dict = {}
    for r in validation_results:
        key = (r["method"], r["graph_mode"], r.get("graph_type", "all"))
        score = r["metrics"][PRIMARY_METRIC]
        if key not in best_by_combo or score > best_by_combo[key]["metrics"][PRIMARY_METRIC]:
            best_by_combo[key] = r

    combos_by_method: Dict = defaultdict(list)
    for key, best_res in best_by_combo.items():
        combos_by_method[key[0]].append((key, best_res))

    # ── Load test data ────────────────────────────────────────────────────
    with open(test_graphs_path, encoding="utf-8") as f:
        all_graphs = json.load(f)

    items = []
    with open(test_dataset_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    if SAMPLE_SIZE:
        items = items[:SAMPLE_SIZE]

    eval_items = [it for it in items if it["id"] in all_graphs]

    # ── Dataset-level metadata (shared across all combos) ─────────────────
    dataset_split_info = {
        "dataset":                os.path.splitext(os.path.basename(test_dataset_path))[0],
        "split":                  "test",
        "source_file":            os.path.basename(test_dataset_path),
        "graphs_file":            os.path.basename(test_graphs_path),
        "total_items_in_file":    len(items),
        "items_with_graphs":      len(eval_items),
        "items_skipped_no_graph": len(items) - len(eval_items),
        "sample_size_cap":        SAMPLE_SIZE,
        "primary_metric":         PRIMARY_METRIC,
    }
    print(f"  {len(eval_items)}/{len(items)} items have graphs "
          f"({len(items) - len(eval_items)} skipped)")

    # ── Load existing master summary (merge, never overwrite) ─────────────
    master_path = os.path.join(RESULTS_DIR, "test_results_by_combo.json")
    if os.path.exists(master_path):
        with open(master_path, encoding="utf-8") as f:
            try:
                final_output = json.load(f)
                print(f"  [MERGE] {len(final_output)} existing entries in master summary")
            except json.JSONDecodeError:
                final_output = {}
    else:
        final_output = {}

    # ══════════════════════════════════════════════════════════════════════
    # METHOD LOOP
    # ══════════════════════════════════════════════════════════════════════
    for method, combinations in combos_by_method.items():
        cfg = MODEL_CONFIGS[method]

        # ── Pre-compute base scores once per method ───────────────────────
        print(f"\n>>> BASE SCORES: {method}")
        item_base_scores: Dict = {}

        if cfg["type"] == "dense":
            model = get_model(method)

            # Collect all unique docs across this dataset
            unique_docs: List[str] = []
            doc_to_id: Dict[str, int] = {}
            for it in eval_items:
                for nd in all_graphs[it["id"]]["nodes"]:
                    txt = nd["title"] + " " + nd["text"]
                    if txt not in doc_to_id:
                        doc_to_id[txt] = len(unique_docs)
                        unique_docs.append(txt)

            # Apply query and doc prefixes (required for BGE, E5, etc.)
            q_prefix = cfg.get("query_prefix", "")
            d_prefix = cfg.get("doc_prefix", "")
            q_texts = [q_prefix + it["question"] for it in eval_items]
            d_texts = [d_prefix + d for d in unique_docs]

            print(f"  Encoding {len(q_texts)} queries + {len(d_texts)} unique docs ...")
            q_emb = model.encode(q_texts, convert_to_tensor=True,
                                 show_progress_bar=True, batch_size=128)
            d_emb = model.encode(d_texts, convert_to_tensor=True,
                                 show_progress_bar=True, batch_size=128)

            for i, it in enumerate(eval_items):
                idxs = [doc_to_id[nd["title"] + " " + nd["text"]]
                        for nd in all_graphs[it["id"]]["nodes"]]
                item_base_scores[it["id"]] = (
                    util.cos_sim(q_emb[i:i+1], d_emb[idxs])[0].cpu().tolist()
                )

        else:  # BM25
            for it in tqdm(eval_items, desc="  BM25 scoring"):
                g = all_graphs[it["id"]]
                corpus = [nd["title"].lower().split() + nd["text"].lower().split()
                          for nd in g["nodes"]]
                bm25 = BM25Okapi(corpus)
                item_base_scores[it["id"]] = (
                    bm25.get_scores(it["question"].lower().split()).tolist()
                )

        # ── Per-combo loop ────────────────────────────────────────────────
        for (combo_method, graph_mode, graph_type), best_res in combinations:
            safe = f"{combo_method}_{graph_mode}_{graph_type}".replace("/", "_")
            pred_file = os.path.join(RESULTS_DIR,
                                     f"test_predictions_{safe}_{dataset_label}.json")

            # Hyperparameter metadata for this combo
            if graph_mode == "bfs":
                s_grid = GRID_BFS
            elif graph_mode == "ge":
                s_grid = GRID_GE
            else:
                s_grid = {}

            hyper_info = {
                "model_id":            cfg.get("model", "N/A (sparse)"),
                "model_type":          cfg.get("type", "sparse"),
                "query_prefix":        cfg.get("query_prefix", ""),
                "doc_prefix":          cfg.get("doc_prefix", ""),
                "graph_type_grid":     GRID_GRAPH_TYPE,
                "search_grid":         s_grid,
                "best_config_applied": best_res.get("config", {}),
            }

            # ── Run evaluation ────────────────────────────────────────────
            print(f"  [RUN ] {safe} | dataset={dataset_label}")
            results = []
            predictions = []

            for it in tqdm(eval_items, desc=f"    {safe}", leave=False):
                graph = all_graphs[it["id"]]
                scores = item_base_scores[it["id"]]
                supp = {i for i, nd in enumerate(graph["nodes"]) if nd["is_supporting"]}

                if graph_mode == "none":
                    ranked = sorted(range(len(scores)),
                                    key=lambda i: scores[i], reverse=True)
                elif graph_mode == "bfs":
                    ranked = bfs_retrieve_with_base(
                        graph, scores, **best_res["config"], graph_type=graph_type)
                else:  # ge
                    ge_scores = compute_ge_embeddings(
                        graph, scores, **best_res["config"], graph_type=graph_type)
                    ranked = sorted(range(len(ge_scores)),
                                    key=lambda i: ge_scores[i], reverse=True)

                results.append(eval_ranking(ranked, supp, len(scores)))

                predictions.append({
                    "id":       it["id"],
                    "question": it["question"],
                    "ranked_results": [
                        {
                            "node_idx":      idx,
                            "title":         graph["nodes"][idx]["title"],
                            "text":          graph["nodes"][idx]["text"],
                            "is_supporting": graph["nodes"][idx]["is_supporting"],
                            "base_score":    round(scores[idx], 4),
                        }
                        for idx in ranked
                    ],
                    "graph_structure": {
                        "nodes": graph["nodes"],
                        "edges": graph.get("edges", []),
                    },
                })

            test_metrics = avg_metrics(results)
            summary = {
                "method":          combo_method,
                "graph_mode":      graph_mode,
                "graph_type":      graph_type,
                "dataset_split":   dataset_split_info,
                "hyperparameters": hyper_info,
                "val_metrics":     best_res["metrics"],
                "test_metrics":    test_metrics,
                "data":            predictions,
            }

            with open(pred_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

            print(f"  [DONE] {safe}_{dataset_label} — "
                  f"Val {PRIMARY_METRIC}: {best_res['metrics'][PRIMARY_METRIC]} → "
                  f"Test {PRIMARY_METRIC}: {test_metrics[PRIMARY_METRIC]}")

            final_output[f"{safe}_{dataset_label}"] = summary

    # ── Save master summary (strip 'data' payload to keep it small) ───────
    clean = {k: {ik: iv for ik, iv in v.items() if ik != "data"}
             for k, v in final_output.items()}
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2)

    ds_count = sum(1 for k in clean if k.endswith(f"_{dataset_label}"))
    print(f"\n[COMPLETE] Master summary → {master_path} "
          f"({len(clean)} total, {ds_count} new for '{dataset_label}')")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Maps: validation_best_combined file → test dataset stem
    val_to_test = {
        "results_2wiki_validation_val.json":
            "2wiki_validation_test",
        "results_hotpotqa_validation_val.json":
            "hotpotqa_validation_test",
        "results_musique_validation_val.json":
            "musique_validation_test",
    }

    for val_file, test_stem in val_to_test.items():
        val_path = os.path.join(RESULTS_DIR, val_file)
        if not os.path.exists(val_path):
            print(f"[SKIP] Validation file not found: {val_path}")
            continue

        test_data   = os.path.join(DATA_DIR,   test_stem + ".jsonl")
        test_graphs = os.path.join(GRAPHS_DIR, test_stem + "_hybrid_graphs.json")

        if not os.path.exists(test_data):
            print(f"[SKIP] Test data not found: {test_data}")
            continue
        if not os.path.exists(test_graphs):
            print(f"[SKIP] Test graphs not found: {test_graphs}")
            continue

        dataset_label = test_stem.split("_")[0]  # "2wiki" / "hotpotqa" / "musique"

        run_test_with_best_config(
            val_path,
            test_data,
            test_graphs,
            dataset_label=dataset_label,
        )


if __name__ == "__main__":
    main()
