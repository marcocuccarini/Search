import json
import os
import glob
from config import RESULTS_DIR, PRIMARY_METRIC


# ============================================================
# LOAD VALIDATION FILES
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# EXTRACT BEST PER (method, graph_mode, graph_type)
# ============================================================

def extract_best_by_combo(all_results):
    best_by_combo = {}

    for r in all_results:
        key = (
            r["method"],
            r["graph_mode"],
            r.get("graph_type", "all")
        )

        score = r["metrics"][PRIMARY_METRIC]

        if key not in best_by_combo:
            best_by_combo[key] = r
        else:
            if score > best_by_combo[key]["metrics"][PRIMARY_METRIC]:
                best_by_combo[key] = r

    return best_by_combo


# ============================================================
# SAVE OUTPUT FILE
# ============================================================

def save_best(dataset_name, best_by_combo):
    output = {}

    for (method, graph_mode, graph_type), r in best_by_combo.items():

        output[f"{method}_{graph_mode}_{graph_type}"] = {
            "method": method,
            "graph_mode": graph_mode,
            "graph_type": graph_type,
            "config": r.get("config", {}),
            "metrics": r.get("metrics", {}),
            "primary_score": r["metrics"].get(PRIMARY_METRIC, None),
        }

    out_path = os.path.join(
        RESULTS_DIR,
        f"validation_best_combined_{dataset_name}.json"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"[SAVED] {out_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    # automatically find validation result files
    files = glob.glob(os.path.join(RESULTS_DIR, "results_*validation*.json"))

    if not files:
        print("No validation result files found!")
        return

    for path in files:

        dataset_name = os.path.basename(path).replace("results_", "").replace(".json", "")

        print(f"\nProcessing: {dataset_name}")

        all_results = load_json(path)

        best_by_combo = extract_best_by_combo(all_results)

        save_best(dataset_name, best_by_combo)


if __name__ == "__main__":
    main()