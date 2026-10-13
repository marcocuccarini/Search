# ─── NEW CONFIGURATION FOR GRAPH TYPES ────────────────────────────────────────
# This allows the grid search to toggle which edge types are "active"


import os, sys 

GRID_GRAPH_TYPE = ["triple", "entity", "both"]

GRID_ALPHA = [0.0, 0.3, 0.5, 0.7, 1] 

# ──────────────────────────────────────────────────────────────────────────────
ENABLED_METHODS = {
    "BM25": True,
    "GTE-base": True,
    "GTE-large": True,
    "BGE-base": True,
    "BGE-large": True,
    "BGE-M3": True,
    "E5-base": True,
    "E5-large": True,
}






MODEL_CONFIGS = {
    "BM25": {"type": "sparse"},

    "GTE-base": {"type": "dense", "model": "thenlper/gte-base"},
    "GTE-large": {"type": "dense", "model": "thenlper/gte-large"},

    "BGE-base": {
        "type": "dense",
        "model": "BAAI/bge-base-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: "
    },
    "BGE-large": {
        "type": "dense",
        "model": "BAAI/bge-large-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: "
    },
    "BGE-M3": {
        "type": "dense",
        "model": "BAAI/bge-m3",
        "query_prefix": "Represent this sentence for searching relevant passages: "
    },

    "E5-base": {
        "type": "dense",
        "model": "intfloat/e5-base-v2",
        "query_prefix": "query: ",
        "doc_prefix": "passage: "
    },
    "E5-large": {
        "type": "dense",
        "model": "intfloat/e5-large-v2",
        "query_prefix": "query: ",
        "doc_prefix": "passage: "
    },
}





# ── BFS GRID (independent ablation) ─────────────────────────────
GRID_BFS = {
    "top_k_bm25": [1, 2, 3],
    "max_depth": [1, 2, 3],
    "branch": [3, 5, 10],
    "alpha": [0.0, 0.3, 0.5, 0.7, 1]
}

# ── GE GRID (independent ablation) ──────────────────────────────
GRID_GE = {
    "ge_alpha": [0.0, 0.1, 0.2, 0.3, 0.5],
    "ge_hops": [1, 2, 3]
}

BFS_CONFIGS = [
    {
        "top_k_bm25": top_k,
        "max_depth": depth,
        "branch": branch,
        "alpha": alpha
    }
    for top_k in GRID_BFS["top_k_bm25"]
    for depth in GRID_BFS["max_depth"]
    for branch in GRID_BFS["branch"]
    for alpha in GRID_ALPHA
]

GE_CONFIGS = [
    {
        "ge_alpha": ge_alpha,
        "ge_hops": ge_hops
    }
    for ge_alpha in GRID_GE["ge_alpha"]
    for ge_hops in GRID_GE["ge_hops"]
]



# ── CONFIGURATION 
HERE        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(HERE, "data")
GRAPHS_DIR  = os.path.join(HERE, "graphs")
RESULTS_DIR = os.path.join(HERE, "results")


SAMPLE_SIZE     = None       # Set to None to run on all items found
PRIMARY_METRIC  = "Recall@5"


os.makedirs(RESULTS_DIR, exist_ok=True)
sys.path.insert(0, ".")