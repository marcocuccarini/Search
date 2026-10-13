import json
import os
import uuid
import datetime
import ollama
from dataclasses import dataclass, field
from enum import Enum

# ==========================================
# 1. CORE CLASSES
# ==========================================

class ResponseType(Enum):
    REFUSAL = "refusal"
    GENERATED = "generated"
    MIXED = "mixed"

@dataclass(kw_only=True)
class LLMResponse:
    prompt_id: str
    raw_text: str = field(default="")
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    response_type: ResponseType = field(default=ResponseType.GENERATED)

class OllamaServer:
    def __init__(self, client):
        self.client = client

    def get_models_list(self):
        try:
            response = self.client.list()
            return [m.model for m in response.models]
        except Exception as e:
            print(f"Error fetching models: {e}")
            return []

    def download_model_if_not_exists(self, targeted_model: str):
        names = [m.split(":")[0] for m in self.get_models_list()]
        base_model = targeted_model.split(":")[0]
        
        if base_model not in names:
            print(f"Pulling model {targeted_model}...")
            self.client.pull(targeted_model)
        else:
            print(f"Model {targeted_model} is ready.")

class OllamaChat:
    USER = 'user'
    ASSISTANT = 'assistant'

    def __init__(self, server: OllamaServer, model: str):
        self.server = server
        self.model = model
        self.messages = []
        self.server.download_model_if_not_exists(model)

    def add_history(self, content: str, role: str):
        self.messages.append({'role': role, 'content': content})

    def clear_history(self):
        self.messages = []

    def send_prompt(self, prompt: str, use_history: bool = True) -> LLMResponse:
        current_msg = {'role': self.USER, 'content': prompt}
        msgs_to_send = (self.messages + [current_msg]) if use_history else [current_msg]
        
        try:
            response = self.server.client.chat(
                model=self.model, 
                messages=msgs_to_send, 
                format='json',
                stream=False
            )
            complete_message = response.get('message', {}).get('content', '').strip()
            return LLMResponse(prompt_id=str(uuid.uuid4()), raw_text=complete_message)
        except Exception as e:
            return LLMResponse(prompt_id=str(uuid.uuid4()), raw_text=f'{{"error": "{str(e)}"}}')

class PromptLibrary:
    """Handles prompt logic for both Question-based and Context-only KG extraction."""
    @staticmethod
    def create_kg_prompt(context: str, question: str = None) -> str:
        if question:
            # Targeted extraction based on the question
            return (
                "You are a Knowledge Graph engineer. "
                f"Based on the specific question: '{question}', analyze this text: '{context}'.\n\n"
                "Task: Extract entities and relationships that help answer the question.\n"
                "Return ONLY a JSON object with 'nodes' and 'edges'."
            )
        else:
            # Global extraction based only on text
            return (
                "You are a Knowledge Graph engineer. "
                f"Analyze the following text: '{context}'.\n\n"
                "Task: Extract all significant entities and their relationships to build a comprehensive Knowledge Graph.\n"
                "Return ONLY a JSON object with 'nodes' and 'edges'."
            )

class KnowledgeDataset:
    def __init__(self, file_path: str, chat_client: OllamaChat, library: PromptLibrary):
        self.chat = chat_client
        self.library = library
        self.data = self._load_jsonl(file_path)
        self.results = {}

    def _load_jsonl(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def run_process(self, use_question: bool = True):
        """
        Processes dataset.
        If use_question=True: KG is focused on the question.
        If use_question=False: KG is built from raw text only.
        """
        mode = "WITH QUESTION" if use_question else "CONTEXT ONLY"
        print(f"--- Running in {mode} mode ---")
        
        total = len(self.data)
        for index, item in enumerate(self.data):
            item_id = item.get("id", "unknown")
            question = item.get("question") if use_question else None
            full_text = " ".join(p["paragraph_text"] for p in item.get("paragraphs", []))

            print(f"[{index + 1}/{total}] ID: {item_id}")
            
            prompt = self.library.create_kg_prompt(full_text, question)
            self.chat.clear_history()
            response = self.chat.send_prompt(prompt)

            try:
                kg_data = json.loads(response.raw_text)
            except:
                kg_data = {"error": "JSON Parsing failed", "raw": response.raw_text}

            self.results[item_id] = {
                "mode": mode,
                "knowledge_graph": kg_data
            }

    def save_report(self, output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
        print(f"Report saved to: {output_path}")

import json
import os
import uuid
import datetime
import ollama
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# ... [Keep all your core classes: ResponseType, LLMResponse, OllamaServer, 
# OllamaChat, PromptLibrary, and KnowledgeDataset exactly as they are] ...

# ==========================================
# 2. EXECUTION LOGIC FOR DIRECTORY PROCESSING
# ==========================================

def process_directory(input_dir: str, output_dir: str, model_name: str):
    # 1. Setup Environment
    client = ollama.Client(host='http://127.0.0.1:11434') 
    server = OllamaServer(client)
    chat_session = OllamaChat(server=server, model=model_name)
    library = PromptLibrary()

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # 2. Iterate through files
    input_path = Path(input_dir)
    # Only looking for .jsonl files as per your Dataset class logic
    files_to_process = list(input_path.glob("*.jsonl"))

    if not files_to_process:
        print(f"No .jsonl files found in {input_dir}")
        return

    print(f"Found {len(files_to_process)} files to process.")

    for file_path in files_to_process:
        print(f"\n{'='*50}")
        print(f"Processing: {file_path.name}")
        print(f"{'='*50}")

        try:
            # Initialize dataset for this specific file
            dataset = KnowledgeDataset(str(file_path), chat_session, library)

            # Run the KG extraction (Toggle use_question here)
            dataset.run_process(use_question=False) 

            # Define output filename: original_name_KG.json
            output_filename = f"{file_path.stem}_KG.json"
            output_full_path = os.path.join(output_dir, output_filename)

            # Save the result
            dataset.save_report(output_full_path)
            
        except Exception as e:
            print(f"Failed to process {file_path.name}: {e}")

if __name__ == "__main__":
    # Configure your paths here
    DATA_DIRECTORY = "data"
    OUTPUT_DIRECTORY = "dataKG_without" # You mentioned saving back into 'data'
    MODEL = "gemma4:latest"

    process_directory(DATA_DIRECTORY, OUTPUT_DIRECTORY, MODEL)