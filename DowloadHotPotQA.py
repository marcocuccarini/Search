import json
import os
from datasets import load_dataset

def save_hotpotqa_to_dati():
    # 1. Configurazione percorsi
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(current_dir, "dati")
    output_path = os.path.join(folder_path, "hotpotqa_validation.jsonl")

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    print(f"Caricamento dataset: HotpotQA...")
    
    # Carichiamo lo split 'validation'. 
    # HotpotQA ha due versioni: 'distractor' (10 paragrafi) e 'fullwiki'. 
    # Usiamo 'distractor' che è il formato standard per i test di retrieval.
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")

    print(f"Record totali trovati: {len(dataset)}")

    with open(output_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(dataset):
            # In HotpotQA, supporting_facts è una lista di [titolo, indice_frase]
            gold_titles = set(item['supporting_facts']['title'])
            
            # 2. Formattazione Paragrafi
            paragraphs = []
            context = item['context'] # Formato: {'title': [...], 'sentences': [[...], [...]]}
            
            for idx, (title, sentences) in enumerate(zip(context['title'], context['sentences'])):
                paragraphs.append({
                    "idx": idx,
                    "title": title,
                    "paragraph_text": "".join(sentences),
                    "is_supporting": title in gold_titles
                })
            
            # 3. Decomposizione (HotpotQA non ha la decomposizione nativa come MuSiQue)
            # Creiamo una struttura compatibile ma vuota o basata sui fatti di supporto
            decomposition = []
            for d_idx, title in enumerate(gold_titles):
                decomposition.append({
                    "id": d_idx + 1,
                    "question": f"Find information about {title}",
                    "answer": "Check supporting facts",
                    "paragraph_support_idx": None
                })

            # 4. Costruzione oggetto finale
            entry = {
                "id": item['id'],
                "paragraphs": paragraphs,
                "question": item['question'],
                "question_decomposition": decomposition,
                "answer": item['answer'],
                "answer_aliases": [], # HotpotQA non fornisce alias
                "answerable": True,
                "level": item['level'], # Campo extra utile: 'easy', 'medium', 'hard'
                "type": item['type']    # Campo extra utile: 'comparison' o 'bridge'
            }
            
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
            if i % 2000 == 0:
                print(f"Processati {i}/{len(dataset)} record...")

    print(f"\n✅ Completato! HotpotQA salvato in: {output_path}")
    print(f"Dimensione finale: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    save_hotpotqa_to_dati()