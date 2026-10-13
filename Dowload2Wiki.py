import json
import os
from datasets import load_dataset

def save_framolfese_to_dati():
    # 1. Configurazione percorsi
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(current_dir, "dati")
    output_path = os.path.join(folder_path, "2wiki_validation.jsonl")

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    print(f"Caricamento dataset: framolfese/2WikiMultihopQA...")
    
    # Questa versione di solito usa 'train' e 'test' (o 'dev')
    # Carichiamo lo split 'validation' (o 'dev' se 'validation' non esiste)
    try:
        dataset = load_dataset("framolfese/2WikiMultihopQA", split="validation")
    except:
        dataset = load_dataset("framolfese/2WikiMultihopQA", split="test")

    print(f"Record totali trovati: {len(dataset)}")

    with open(output_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(dataset):
            # Mappatura specifica per la versione framolfese
            # Nota: In questa versione i campi potrebbero essere già processati
            
            gold_titles = set(item.get('supporting_facts', {}).get('title', []))
            
            # Gestione Context
            paragraphs = []
            context = item.get('context', {})
            
            # Se context è un dizionario con 'title' e 'sentences' (comune in questa versione)
            if isinstance(context, dict) and 'title' in context:
                for idx, (title, sentences) in enumerate(zip(context['title'], context['sentences'])):
                    # Alcune versioni hanno sentences come lista di liste, altre come lista di stringhe
                    p_text = " ".join(sentences) if isinstance(sentences, list) else sentences
                    paragraphs.append({
                        "idx": idx,
                        "title": title,
                        "paragraph_text": p_text,
                        "is_supporting": title in gold_titles
                    })
            
            # Decomposizione (Evidences)
            decomposition = []
            evidences = item.get('evidences', [])
            # Se evidences è un dizionario (formato dataset framolfese)
            if isinstance(evidences, dict):
                # Solitamente le triple sono in una lista sotto una chiave specifica
                ev_list = evidences.get('fact', [])
            else:
                ev_list = evidences

            for d_idx, ev in enumerate(ev_list):
                if len(ev) >= 3:
                    decomposition.append({
                        "id": d_idx + 1,
                        "question": f"{ev[0]} >> {ev[1]}",
                        "answer": ev[2],
                        "paragraph_support_idx": None
                    })

            # Costruzione riga finale
            entry = {
                "id": item.get('_id') or item.get('id') or f"idx_{i}",
                "paragraphs": paragraphs,
                "question": item.get('question', ''),
                "question_decomposition": decomposition,
                "answer": item.get('answer', ''),
                "answer_aliases": [],
                "answerable": True
            }
            
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
            if i % 1000 == 0:
                print(f"Processati {i}/{len(dataset)} record...")

    print(f"\n✅ Completato! File salvato in: {output_path}")
    print(f"Dimensione finale: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    save_framolfese_to_dati()