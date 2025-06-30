import os
import json
from typing import List, Dict, Optional
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

from retriever.chromadb_index import search

# ✅ Konfigurasi Gemini 2.0 Flash
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm = genai.GenerativeModel("gemini-2.0-flash")

# ✅ Prompt Template
BASE_PROMPT = """You are an expert assistant. Below are information chunks retrieved from multiple video transcripts and image links that may help answer the user's question.

Your task is to analyze these snippets and image references to provide a clear, concise, and informative answer to the user. Consider both text and image context. If image URLs are included, assume that the image is relevant and try to infer its meaning from the text and surrounding context. Prioritize accurate and reasoned answers over speculation. If multiple answers are possible, summarize or compare them.

[QUESTION]
{query}

"""

# ✅ Prompt builder
def build_prompt(query: str, contexts: List[Dict]) -> str:
    prompt = BASE_PROMPT.format(query=query)

    for i, ctx in enumerate(contexts):
        text = ctx.get("text", "(no text)")
        meta = ctx.get("metadata", {})
        source = meta.get("source_id", "unknown")
        image_url = meta.get("image_url", None)

        prompt += f"\n[CONTEXT {i+1}]\n"
        prompt += f"- Text: {text.strip()[:200]}...\n"
        prompt += f"- Source: {source}\n"
        if image_url:
            prompt += f"- Image URL: {image_url}\n"

    return prompt

# ✅ Agentic RAG
def run_agentic_rag(query: str, n_chunks: int = 15) -> Dict:
    results = search(query, n_results=n_chunks)
    contexts = []

    for doc, meta, dist in zip(results["documents"], results["metadatas"], results["distances"]):
        contexts.append({
            "text": doc,
            "metadata": meta,
            "distance": dist
        })

    if not contexts:
        return {
            "query": query,
            "answer": "Sorry, I couldn't find any relevant information to answer that.",
            "used_contexts": []
        }

    prompt = build_prompt(query, contexts)
    print("📝 Prompt sent to Gemini:\n", prompt[:500], "...")  # Debug cut-off
    response = llm.generate_content(prompt)

    return {
        "query": query,
        "answer": response.text,
        "used_contexts": contexts,
        "timestamp": datetime.now().isoformat()
    }
