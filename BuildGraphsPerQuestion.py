import json
import os
import glob
import traceback

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR  = os.path.join(_HERE, "data")
KG_DIR     = os.path.join(_HERE, "dataKG")
OUTPUT_DIR = os.path.join(_HERE, "graphs")
# ──────────────────────────────────────────────────────────────────────────────

def _entity_in_text(entity, text) -> bool:
    """Case-insensitive check if entity exists in text or title."""
    if entity is None or text is None:
        return False
    return str(entity).lower() in str(text).lower()

def _get_triplet(kg_edge):
    """Standardizes KG edges into (subject, relation, object)."""
    if isinstance(kg_edge, list):
        try:
            s = kg_edge[0]
            r = kg_edge[1] if len(kg_edge) > 1 else "connected_to"
            o = kg_edge[2] if len(kg_edge) > 2 else ""
            return str(s), str(r), str(o)
        except: return None
    elif isinstance(kg_edge, dict):
        s = kg_edge.get("source") or kg_edge.get("entity") or ""
        r = kg_edge.get("source_relationship") or kg_edge.get("relation") or "connected_to"
        o = kg_edge.get("target") or kg_edge.get("entity2") or ""
        return str(s), str(r), str(o)
    return None

def build_hybrid_pseudo_graph(item_id, item_data, kg_item):
    paragraphs = item_data.get("paragraphs", [])
    nodes = []
    for p in paragraphs:
        nodes.append({
            "idx": p.get("idx"),
            "title": p.get("title", ""),
            "text": p.get("paragraph_text", ""),
            "is_supporting": p.get("is_supporting", False)
        })

    # Extract all triplets and the set of all unique entities in the KG
    triplets = []
    unique_entities = set()
    if kg_item:
        kg = kg_item.get("knowledge_graph", kg_item)
        raw_edges = kg.get("edges", [])
        for e in raw_edges:
            t = _get_triplet(e)
            if t and t[0] and t[2]:
                triplets.append(t)
                unique_entities.add(t[0])
                unique_entities.add(t[2])

    final_edges = []
    seen_kg_links = set()

    # --- TYPE 1: KG-BRIDGE RELATIONS (S -> O) ---
    for s, r, o in triplets:
        paras_with_s = [n["idx"] for n in nodes if _entity_in_text(s, n["text"]) or _entity_in_text(s, n["title"])]
        paras_with_o = [n["idx"] for n in nodes if _entity_in_text(o, n["text"]) or _entity_in_text(o, n["title"])]

        for i in paras_with_s:
            for j in paras_with_o:
                if i == j: continue
                edge = {
                    "src": i, "dst": j,
                    "relation_type": "kg_bridge",
                    "relation": r,
                    "subject": s, "object": o
                }
                final_edges.append(edge)
                # Keep track to avoid redundant overlap edges for the same pair/entity
                seen_kg_links.add((i, j, s))
                seen_kg_links.add((j, i, o))

    # --- TYPE 2: ENTITY-OVERLAP RELATIONS (Shared Entity) ---
    for entity in unique_entities:
        paras_with_ent = [n["idx"] for n in nodes if _entity_in_text(entity, n["text"]) or _entity_in_text(entity, n["title"])]
        
        # Connect all paragraphs that share this entity
        for i in range(len(paras_with_ent)):
            for j in range(i + 1, len(paras_with_ent)):
                idx_a, idx_b = paras_with_ent[i], paras_with_ent[j]
                
                # Only add as 'overlap' if it's not already fulfilling a specific KG bridge for this entity
                final_edges.append({
                    "src": idx_a, "dst": idx_b,
                    "relation_type": "entity_overlap",
                    "shared_entity": entity
                })
                # Since it's a shared entity, it's bidirectional
                final_edges.append({
                    "src": idx_b, "dst": idx_a,
                    "relation_type": "entity_overlap",
                    "shared_entity": entity
                })

    return {
        "id": item_id,
        "question": item_data.get("question", ""),
        "answer": item_data.get("answer", ""),
        "nodes": nodes,
        "edges": final_edges
    }

def process_file(jsonl_path, kg_path):
    stem = os.path.splitext(os.path.basename(jsonl_path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{stem}_hybrid_graphs.json")
    kg_map = {}
    if os.path.exists(kg_path):
        with open(kg_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            if isinstance(raw, list):
                for entry in raw:
                    eid = entry.get("id") or entry.get("question_id")
                    if eid: kg_map[eid] = entry
            else:
                kg_map = raw

    results = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            iid = item.get("id")
            if iid: results[iid] = build_hybrid_pseudo_graph(iid, item, kg_map.get(iid))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Done: {stem}")

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    for f in sorted(glob.glob(os.path.join(INPUT_DIR, "*.jsonl"))):
        kg_f = os.path.join(KG_DIR, os.path.splitext(os.path.basename(f))[0] + "_KG.json")
        process_file(f, kg_f)

if __name__ == "__main__":
    main()