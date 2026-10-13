# Search+ revision experiments

Copy this `revision/` folder into the root of the Search+ repository (next to
`config.py`, `data/`, `dataKG/`, `dataKG_without/`, `graphs/`). Everything uses
the repo's own code: the BFS / NSP ("ge") functions and metrics are verbatim
copies from `TestExperiments.py` (`core.py`), graphs are built with
`BuildGraphsPerQuestion.build_hybrid_pseudo_graph`, configs are tuned on the
validation split with `config.BFS_CONFIGS` / `GE_CONFIGS` and
`config.PRIMARY_METRIC`, exactly like `ValidateExperiments.py`.

All outputs go to `revision_results/`.

## Run order

```bash
pip install numpy rank_bm25 matplotlib networkx          # + torch, sentence-transformers for step 1

# 1. Dense retriever scores, once (GPU). BM25 is computed on the fly.
python revision/compute_base_scores.py                     # all 7 dense models, val + test
#    or just the main one:  --methods BGE-large

# 2. (optional) Compare with the paper's appendix tables
python revision/reproduce.py

# 3. Ablation + query-unaware graph + random graph   (main experiment)
python revision/run_experiments.py                         # all retrievers with scores
python revision/run_experiments.py --methods BM25 BGE-large   # faster

# 4. KG quality and its link to the reranking gain   (needs step 3)
python revision/kg_quality.py --methods BM25 BGE-large

# 5. Similar methods on the same graphs: PPR (HippoRAG / GraphER baseline), GCS (GraphER)
python revision/similar_methods.py --methods BM25 BGE-large

# 6. Bad-KG example (needs step 3), one per dataset/retriever you want
python revision/bad_example.py --dataset musique --method BGE-large
python revision/bad_example.py --dataset musique --method BGE-large --pick 1   # next candidate

# 7. Cost: Search+ (query-aware / query-unaware) vs list-wise LLM ranking (needs Ollama running)
python revision/graph_latency.py --n 100     # after step 8: list-wise timings are read from llm_rerank.py

# 8. List-wise LLM reranking with the same protocol as Search+ (all questions, all passages,
#    full text, retriever order, failures counted with retriever fallback)
python revision/llm_rerank.py --dataset musique --retriever BGE-large

# 9. (optional, slow, separate venv) HippoRAG 2 on the same candidate pools
python revision/hipporag2_baseline.py --dataset musique --limit 20
```

Runtime on CPU per retriever: step 3 about 15 min for the three datasets,
steps 4–6 a few minutes. Step 1 needs the GPU.

## What each step produces

| Step | Files | For the paper |
|---|---|---|
| 3 | `experiments/<ds>_<retriever>.csv` — test metrics for every graph variant (`aware`, `unaware`, `unaware_norm`, `random`) × method (`bfs`, `ge`) × edge type, with the val-tuned config; random = mean ± std over 5 seeds | ablation table, query-unaware and random-graph table |
| 3 | `significance/<ds>_<retriever>.csv` — paired bootstrap: ablation (triple/entity vs both), each variant vs aware, each variant vs baseline | significance markers |
| 4 | `kg_quality/summary.csv` — quality of the query-aware vs query-unaware KG; `quality_vs_gain_correlation.csv`; `gain_by_quality_group.csv`; per-query CSVs | KG-quality section |
| 5 | `similar_methods/<ds>_<retriever>.csv` + significance vs BFS/NSP | comparison with similar methods |
| 7 | `latency/summary.csv`, `latency/latency_table.tex` — per-query KG extraction and graph-build time on the same random questions for both modes | latency (point 8) |
| 8 | `llm_rerank/summary.csv` — retriever vs LLM vs Search+ on all test questions, valid / incomplete / invalid / error rates, latency, prompt tokens; `significance.csv` Search+ vs LLM | list-wise comparison (point 1) |
| 6 | `bad_examples/<ds>_<retriever>.json` (all hurt queries, with diagnosed causes) and, for the chosen one, `.png/.pdf` graph, `.tex` ranking table + passages, `_kg.json` | bad-KG example |

### Graph variants
* `aware` — your paper graphs (`graphs/*_hybrid_graphs.json`).
* `unaware` — built from `dataKG_without/` with your builder.
* `unaware_norm` — same, with `_` replaced by spaces in entity names: about 31%
  of query-unaware entities are ids like `Clara_Peeters` that the substring
  matcher can never find, so this variant makes the comparison fair.
* `random` — every connected passage pair of the aware graph moved to a random
  pair: same number of connections, multiplicities and edge types.

### KG quality measures (`kg_quality.py` docstring has the definitions)
parse_ok, n_triples, grounding (entities found in the passages), hub_share
(generic entities), gold_title_recall, bridge_recall (intermediate answers from the
MuSiQue / 2Wiki decompositions), gold_pair_recall, gold_connected, edge_precision,
distractor_weight, gold_isolated.

## Notes found while building this
* `recall_at_k` computes the **proportion** of gold passages in the top k. So
  Section 2.1.1 of the paper is right; Section 6.1 and the repo README
  ("≥ 1 gold passage") are wrong.
* `BuildGraphsPerQuestion.py` iterates over a Python `set`, so the edge order of
  a rebuilt graph changes between runs (it only affects BFS tie-breaking,
  ≤ 0.005 FR@5). The scripts fix `PYTHONHASHSEED=0`.
* The builder reads the relation from `relation` / `source_relationship` only;
  most KG edges use `relationship`, so the stored relation is `connected_to`.
  This does not change the graphs, only the edge labels.
* The substring matcher also matches inside words (e.g. `NY` in other words,
  `Location` in `Locations`); `bad_example.py` flags these as `substring`.
* With your code and data, the BM25 baselines differ from the appendix tables by
  1–3 queries and BFS comes out 0.1–2.6 points higher; NSP matches within
  0.0025. Report all new tables from one consistent run (step 3), including
  the `aware` rows, rather than mixing with the old appendix numbers.
