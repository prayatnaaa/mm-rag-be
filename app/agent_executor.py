import os
import hashlib
from typing import List, Dict, Optional, Union
from datetime import datetime
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

from app.retriever.chromadb_index import search

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm = genai.GenerativeModel("gemini-2.0-flash")


BASE_PROMPT = """You are an expert assistant. Below are information chunks retrieved from multiple video transcripts and image links that may help answer the user's question.

Your task is to analyze these snippets and image references to provide a clear, concise, and informative answer to the user. Consider both text and image context. If image URLs are included, assume that the image is relevant and try to infer its meaning from the text and surrounding context. Prioritize accurate and reasoned answers over speculation. If multiple answers are possible, summarize or compare them.

[QUESTION]
{query}

"""

def deduplicate_results(docs: List[str], metas: List[dict], dists: List[float]):
    seen_hashes = set()
    deduped = []

    for doc, meta, dist in zip(docs, metas, dists):
        if isinstance(meta, dict):
            clean_meta = meta
        elif isinstance(meta, list) and meta and isinstance(meta[0], dict):
            clean_meta = meta[0]
        else:
            clean_meta = {"raw_meta": str(meta)}  # fallback

        # Gunakan isi teks untuk deduplication
        normalized_text = doc.strip().lower()
        text_hash = hashlib.md5(normalized_text.encode("utf-8")).hexdigest()

        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            deduped.append({
                "text": doc,
                "metadata": clean_meta,
                "distance": dist
            })

    return deduped

def build_prompt(query: str, contexts: List[Dict]) -> str:
    prompt = BASE_PROMPT.format(query=query)

    for i, ctx in enumerate(contexts):
        raw_text = ctx.get("text", "(no text)")
        # pastikan text adalah string
        text = raw_text if isinstance(raw_text, str) else str(raw_text)
        meta = ctx["metadata"]
        source = meta.get("source_id", "unknown")
        image_url = meta.get("image_url")

        prompt += f"\n[CONTEXT {i+1}]\n"
        prompt += f"- Text: {text.strip()[:200]}...\n"
        prompt += f"- Source: {source}\n"
        if image_url:
            prompt += f"- Image URL: {image_url}\n"

    return prompt

def run_agentic_rag(
    query: str,
    image: Optional[Union[str, Image.Image]] = None,
    n_chunks: int = 10  # top-N per modality
) -> Dict:
    # 1) Search multimodal
    result = search(query=query, image=image, n_results=50)  # ambil lebih banyak utk filtering

    # 2) Ambil text & image data
    text_docs   = result["text_results"]["documents"][0]
    text_metas  = result["text_results"]["metadatas"][0]
    text_dists  = result["text_results"]["distances"][0]

    image_docs  = result["image_results"]["documents"][0]
    image_metas = result["image_results"]["metadatas"][0]
    image_dists = result["image_results"]["distances"][0]

    # 3) Ambil top-N berdasarkan distance
    top_text = sorted(zip(text_docs, text_metas, text_dists), key=lambda x: x[2])[:n_chunks]
    top_image = sorted(zip(image_docs, image_metas, image_dists), key=lambda x: x[2])[:n_chunks]

    all_docs  = [x[0] for x in top_text + top_image]
    all_metas = [x[1] for x in top_text + top_image]
    all_dists = [x[2] for x in top_text + top_image]

    # 4) Deduplicate & build prompt
    contexts = deduplicate_results(all_docs, all_metas, all_dists)

    if not contexts:
        return {
            "query": query,
            "answer": "Sorry, I couldn't find any relevant information to answer that.",
            "used_contexts": []
        }

    prompt = build_prompt(query, contexts)
    print("📝 Prompt sent to Gemini:\n", prompt[:500], "...\n")
    response = llm.generate_content(prompt)

    return {
        "query": query,
        "answer": response.text,
        "used_contexts": contexts,
    }