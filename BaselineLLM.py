import json
import os
import requests
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "data"
GRAPH_DIR = "graphs"
OUT_DIR = "results"

os.makedirs(OUT_DIR, exist_ok=True)

OLLAMA_URL = "https://o24lj0zhgnldic-11434.proxy.runpod.net/api/generate"
MODEL_NAME = "gemma4:latest"
TIMEOUT = 600

# ============================================================
# LLM CALL
# ============================================================

def call_llm(prompt):
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 1024
                }
            },
            timeout=TIMEOUT
        )
        r.raise_for_status()
        return r.json().get("response", "")
    except Exception as e:
        print("LLM error:", e)
        return ""

# ============================================================
# PROMPT (STRICT RANKING)
# ============================================================

def build_prompt(question, nodes):
    docs = "\n".join(
        f"[{i}] {n.get('title','')} {n.get('text','')[:200]}"
        for i, n in enumerate(nodes)
    )

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

# ============================================================
# PARSER (JSON ONLY)
# ============================================================

def parse_rank(text, n):
    try:
        ranked = json.loads(text.strip())

        if not isinstance(ranked, list):
            raise ValueError()

        cleaned = []
        seen = set()

        for x in ranked:
            x = int(x)
            if 0 <= x < n and x not in seen:
                cleaned.append(x)
                seen.add(x)

        for i in range(n):
            if i not in seen:
                cleaned.append(i)

        return cleaned[:n]

    except:
        return list(range(n))

# ============================================================
# RESUME SUPPORT
# ============================================================

def load_done_ids(path):
    done = set()

    if not os.path.exists(path):
        return done

    with open(path, "r") as f:
        for line in f:
            try:
                done.add(json.loads(line)["id"])
            except:
                pass

    return done

# ============================================================
# RUN LAYER (CHECKPOINTED EVERY 50)
# ============================================================

def run_model(graphs, items, out_file, save_every=50):

    done_ids = load_done_ids(out_file)

    mode = "a" if os.path.exists(out_file) else "w"

    buffer = []

    with open(out_file, mode) as f:

        for it in tqdm(items, desc="RUN"):

            qid = it["id"]

            # skip already done
            if qid in done_ids:
                continue

            question = it["question"]
            nodes = graphs[qid]["nodes"]

            # attach ground truth relevance (NOT used in inference)
            relevance = [
                {
                    "doc_id": i,
                    "is_supporting": bool(n.get("is_supporting", False))
                }
                for i, n in enumerate(nodes)
            ]

            # LLM inference
            prompt = build_prompt(question, nodes)
            response = call_llm(prompt)

            ranked = parse_rank(response, len(nodes))

            result = {
                "id": qid,
                "question": question,
                "ranked": ranked,
                "relevance": relevance
            }

            buffer.append(result)

            # checkpoint save every 50
            if len(buffer) >= save_every:
                for r in buffer:
                    f.write(json.dumps(r) + "\n")
                f.flush()
                buffer = []

        # final flush
        for r in buffer:
            f.write(json.dumps(r) + "\n")
        f.flush()

# ============================================================
# MAIN PIPELINE
# ============================================================

def run():

    datasets = [f for f in os.listdir(DATA_DIR) if f.endswith(".jsonl")]

    summary = {}

    for file in datasets:

        stem = file.replace(".jsonl", "")

        graph_path = os.path.join(GRAPH_DIR, stem + "_hybrid_graphs.json")
        data_path = os.path.join(DATA_DIR, file)

        if not os.path.exists(graph_path):
            continue

        print("\n==============================")
        print("DATASET:", stem)
        print("==============================")

        graphs = json.load(open(graph_path))
        items = [json.loads(l) for l in open(data_path)]

        eval_items = [x for x in items if x["id"] in graphs]

        out_file = os.path.join(OUT_DIR, f"{stem}_run.jsonl")

        run_model(graphs, eval_items, out_file, save_every=50)

        summary[stem] = {
            "file": out_file
        }

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run()