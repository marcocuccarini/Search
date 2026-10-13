import json
import os
import random

def split_all_files_in_folder(folder_path="data", val_size=2000, test_size=2000, seed=42):
    """
    Scansiona la cartella e splitta ogni file .jsonl trovato.
    """
    random.seed(seed)
    
    # Verifica che la cartella esista
    if not os.path.exists(folder_path):
        print(f"Errore: La cartella '{folder_path}' non esiste.")
        return

    # Ottieni tutti i file .jsonl (escludendo quelli già splittati precedentemente)
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.jsonl') 
                 and not (f.endswith('_val.jsonl') or f.endswith('_test.jsonl'))]

    if not all_files:
        print(f"Nessun file .jsonl originale trovato in '{folder_path}'.")
        return

    print(f"Trovati {len(all_files)} file da processare.")

    for file_name in all_files:
        full_input_path = os.path.join(folder_path, file_name)
        base_name = os.path.splitext(file_name)[0]
        
        # Lettura file
        with open(full_input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Shuffle per garantire la casualità
        random.shuffle(lines)
        num_total = len(lines)
        
        # Logica di split
        if num_total >= (val_size + test_size):
            val_data = lines[:val_size]
            test_data = lines[val_size : val_size + test_size]
        else:
            mid = num_total // 2
            val_data = lines[:mid]
            test_data = lines[mid:]
        
        # Scrittura dei due nuovi file
        val_filename = f"{base_name}_val.jsonl"
        test_filename = f"{base_name}_test.jsonl"
        
        with open(os.path.join(folder_path, val_filename), 'w', encoding='utf-8') as f_val:
            f_val.writelines(val_data)
            
        with open(os.path.join(folder_path, test_filename), 'w', encoding='utf-8') as f_test:
            f_test.writelines(test_data)
            
        print(f"---")
        print(f"File: {file_name}")
        print(f"  - Record originali: {num_total}")
        print(f"  - Creato {val_filename} ({len(val_data)} record)")
        print(f"  - Creato {test_filename} ({len(test_data)} record)")

    print("\n✅ Elaborazione di tutti i file completata.")

if __name__ == "__main__":
    split_all_files_in_folder()