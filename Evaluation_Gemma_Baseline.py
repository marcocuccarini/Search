import json
import os

def calculate_full_recall_performance(file_path, top_k_list=[3, 5, 10]):
    """
    Analizza un file di run JSONL e calcola Recall@K e FullRec@K 
    sia per la baseline originale sia applicando il filtro sui nodi non connessi.
    """
    if not os.path.exists(file_path):
        print(f"File non trovato: {file_path}")
        return None

    # Inizializziamo i dizionari per accumulare i successi per ogni K richiesto
    metrics_summary = {
        k: {
            "orig_recall_hits": 0, "orig_full_rec_hits": 0,
            "filt_recall_hits": 0, "filt_full_rec_hits": 0
        } for k in top_k_list
    }
    
    total_instances = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            ranked_indices = item.get('ranked', [])
            relevance_list = item.get('relevance', [])
            
            # Estraiamo l'insieme di TUTTI i doc_id che sono passaggi Gold (supporting)
            gold_docs = {r['doc_id'] for r in relevance_list if r.get('is_supporting', False)}
            if not gold_docs:
                continue  # Salta le istanze senza gold ground truth
                
            total_instances += 1
            
            # --- SCENARIO 1: ORIGINALE (Baseline Semantica) ---
            for k in top_k_list:
                original_top_k = ranked_indices[:k]
                original_hits = set(original_top_k).intersection(gold_docs)
                
                # Recall standard: almeno uno presente
                if len(original_hits) > 0:
                    metrics_summary[k]["orig_recall_hits"] += 1
                # FULL RECALL: tutti i passaggi necessari devono essere inclusi nella Top K
                if len(original_hits) == len(gold_docs):
                    metrics_summary[k]["orig_full_rec_hits"] += 1

            # --- SCENARIO 2: FILTRATO (Simulazione Rimozione Nodi senza collegamenti) ---
            # I nodi non connessi vengono identificati e retrocessi in fondo alla coda
            connected_passages = []
            unconnected_passages = []
            
            for doc_id in ranked_indices:
                # Sostituisci questo True con la condizione reale del tuo grafo (es: doc_id in active_edges)
                # Se un nodo della Top 3 iniziale non ha collegamenti, diventa False
                is_connected = True 
                
                if is_connected:
                    connected_passages.append(doc_id)
                else:
                    unconnected_passages.append(doc_id)
            
            new_ranked = connected_passages + unconnected_passages
            
            # Calcolo metriche post-filtraggio strutturale
            for k in top_k_list:
                filtered_top_k = new_ranked[:k]
                filtered_hits = set(filtered_top_k).intersection(gold_docs)
                
                if len(filtered_hits) > 0:
                    metrics_summary[k]["filt_recall_hits"] += 1
                if len(filtered_hits) == len(gold_docs):
                    metrics_summary[k]["filt_full_rec_hits"] += 1

    # Output dei risultati formattati
    print(f"\n=================================================================")
    print(f" REPORT METRICHE COMPRENSIVO: {os.path.basename(file_path)}")
    print(f"=================================================================")
    print(f"Totale istanze utili elaborate: {total_instances}\n")
    
    for k in top_k_list:
        orig_rec = metrics_summary[k]["orig_recall_hits"] / total_instances
        orig_frec = metrics_summary[k]["orig_full_rec_hits"] / total_instances
        filt_rec = metrics_summary[k]["filt_recall_hits"] / total_instances
        filt_frec = metrics_summary[k]["filt_full_rec_hits"] / total_instances
        
        print(f"--- Classifica a TOP {k} ---")
        print(f"  [Baseline]  Recall@{k}: {orig_rec:.4f}  |  Full Recall@{k}: {orig_frec:.4f}")
        print(f"  [Filtrato]  Recall@{k}: {filt_rec:.4f}  |  Full Recall@{k}: {filt_frec:.4f}")
        print(f"  Delta (Δ):  Recall:  {filt_rec-orig_rec:+.4f}  |  Full Recall:  {filt_frec-orig_frec:+.4f}\n")

# --- ESECUZIONE SUI TUOI FILE ---
calculate_full_recall_performance("results/2wiki_validation_test_run.jsonl")
calculate_full_recall_performance("results/musique_validation_test_run.jsonl")
calculate_full_recall_performance("results/hotpotqa_validation_test_run.jsonl")