import os
import json
import requests
from dotenv import load_dotenv
import re
from difflib import SequenceMatcher


load_dotenv()
import google.generativeai as genai
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

from app.rag_pipeline import run_rag_pipeline
from app.db.metadata_store import get_active_sources

# ====================================
# Gemini model setup
# ====================================
model = genai.GenerativeModel("gemini-2.0-flash")

# ====================================
# Tools
# ====================================

def list_sources():
    """Return list of active source IDs."""
    sources = get_active_sources()
    return {"sources": list(sources.keys())}


def rag_query(query: str, image_path: str = None):
    """Run full multimodal RAG pipeline."""
    return run_rag_pipeline(query, image_path=image_path)

def extract_source_hint(query: str):
    """Coba temukan source_id berdasarkan kemiripan judul."""
    sources = get_active_sources()
    best_match = None
    highest_score = 0.0
    for source_id, meta in sources.items():
        title = meta.get("title", "").lower()
        if not title:
            continue
        score = SequenceMatcher(None, query.lower(), title).ratio()
        if score > highest_score and score > 0.6:
            best_match = source_id
            highest_score = score
    return best_match



# ====================================
# Tool descriptions for prompting
# ====================================

tool_descriptions = {
    "list_sources": "Mengembalikan daftar sumber aktif yang tersedia.",
    "rag_query": "Menjawab pertanyaan user berdasarkan isi video dan/atau gambar. Argumen: query (str), image_path (opsional, path ke gambar lokal).",
}


# ====================================
# Prompt builder
# ====================================

def build_agent_prompt(user_query: str):
    """Generate prompt to let Gemini choose which function to use."""
    prompt = f"""
Kamu adalah asisten cerdas untuk menjawab pertanyaan berdasarkan video dan gambar. Berikut adalah daftar fungsi yang dapat kamu gunakan:

{json.dumps(tool_descriptions, indent=2)}

Pertanyaan user:
\"\"\"{user_query}\"\"\"

Tentukan fungsi yang paling relevan untuk menjawab pertanyaan tersebut. Balas hanya dalam format JSON berikut tanpa tambahan teks apapun:

{{
  "function": "nama_fungsi",
  "arguments": {{
    ...
  }}
}}

Jika tidak tahu, gunakan fungsi 'rag_query' sebagai default.
"""
    return prompt


# ====================================
# Agent Executor
# ====================================

def clean_json_response(text):
    cleaned = re.sub(r"^```json\s*", "", text)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()

def AgentExecutor(user_query: str, image_path: str = None):
    prompt = build_agent_prompt(user_query)
    response = model.generate_content(prompt)
    print("[DEBUG] Function call response:", response.text)

    try:
        cleaned_text = clean_json_response(response.text)
        print("[DEBUG] Cleaned response:", cleaned_text)
        parsed = json.loads(cleaned_text)
        fn_name = parsed.get("function")
        args = parsed.get("arguments", {})


        # Inject image path if applicable
        if image_path:
            args["image_path"] = image_path

        func_map = {
            "list_sources": list_sources,
            "rag_query": rag_query,
        }

        if fn_name not in func_map:
            return {"error": f"Fungsi tidak dikenali: {fn_name}", "raw_response": response.text}

        result = func_map[fn_name](**args)
        return {"function": fn_name, "result": result}

    except Exception as e:
        return {"error": str(e), "raw_response": response.text}
