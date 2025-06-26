import os
import hashlib
import requests
from typing import Optional
from dotenv import load_dotenv
from collections import defaultdict

from app.retriever.embed_clip import embed_text_image
from app.retriever.faiss_index import search_similar_chunks
from app.db.metadata_store import get_active_sources, get_chunk_counts_per_source
from sentence_transformers import CrossEncoder
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_results(query, contexts):
    texts = [(query, ctx["text"]) for ctx in contexts if "text" in ctx]
    scores = reranker.predict(texts)
    for ctx, score in zip(contexts, scores):
        ctx["relevance_score"] = float(score)
    return sorted(contexts, key=lambda x: -x["relevance_score"])


def compute_source_weights(chunk_counts: dict[str, int], exponent: float = 1.0) -> dict[str, float]:
    if not chunk_counts:
        return {}
    
    total = sum(chunk_counts.values())
    weights = {}
    for source_id, count in chunk_counts.items():
        ratio = count / total
        weight = 1.0 / (ratio ** exponent)
        weights[source_id] = weight

    max_weight = max(weights.values())
    return {k: v / max_weight for k, v in weights.items()}


def deduplicate_contexts(contexts):
    seen = set()
    unique = []
    for ctx in contexts:
        key = hashlib.md5((ctx.get("text", "") + ctx.get("image_url", "")).encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(ctx)
    return unique


def generate_answer_from_gemini(contexts, query: str):
    parts = []
    parts.append("Berikut ini adalah potongan transkrip video dan gambar. Jawablah pertanyaan user secara jelas dan ringkas.")
    parts.append(f"[PERTANYAAN USER]\n{query}")

    for ctx in contexts:
        text = ctx.get("text", "")
        image_url = ctx.get("image_url")
        if image_url:
            try:
                response = requests.get(image_url)
                if response.status_code == 200:
                    image_bytes = response.content
                    parts.append({
                        "mime_type": "image/jpeg",
                        "data": image_bytes
                    })
            except Exception as e:
                print(f"❌ Gagal unduh gambar: {image_url} — {e}")

        if text:
            parts.append(f"[KONTEKS VIDEO]\n{text}")

    response = model.generate_content(parts)
    return {"answer": response.text, "evidence": contexts}


def run_rag_pipeline(query: str, image_path: Optional[str] = None):
    active_sources = get_active_sources()
    active_ids = list(active_sources.keys())
    print("✅ Active source IDs:", active_ids)

    # Step 1: Embedding
    vecs = embed_text_image(query, image_path)
    vec_text = vecs["text_embedding"]
    vec_image = vecs["image_embedding"]

    # Step 2: Retrieval
    results_text = search_similar_chunks(vec_text, top_k=20, allowed_source_ids=active_ids)
    results_image = search_similar_chunks(vec_image, top_k=20, allowed_source_ids=active_ids)
    print(f"🔍 Retrieved: {len(results_text)} text chunks | {len(results_image)} image chunks")

    # Step 3: Filtering
    threshold_text = 0.75
    threshold_image = 0.75
    filtered_text = [r for r in results_text if r["distance"] <= threshold_text]
    filtered_image = [r for r in results_image if r["distance"] <= threshold_image]

    combined = deduplicate_contexts(results_text + results_image)

    # Fallback: top-1 from each
    if not combined:
        fallback_text = results_text[:1] if results_text else []
        fallback_image = results_image[:1] if results_image else []
        combined = deduplicate_contexts(fallback_text + fallback_image)
        print("⚠️ Fallback mode activated due to low similarity.")

    # Step 4: Compute dynamic weights
    chunk_counts = get_chunk_counts_per_source(source_ids=active_ids)  # {source_id: chunk_count}
    source_weights = compute_source_weights(chunk_counts, exponent=1.0)
    print("🎯 Source weights:", source_weights)

    # Step 5: Re-rank with source weight
    combined = sorted(
        combined,
        key=lambda x: x["distance"] / source_weights.get(x["source_id"], 1.0)
    )[:20]

    combined = rerank_results(query, combined)[:20]

    print(f"📎 Final contexts used: {len(combined)}")

    # Step 6: Generate final answer
    if not combined:
        return {
            "answer": "Maaf, saya tidak menemukan informasi yang relevan untuk menjawab pertanyaan atau menjelaskan gambar ini.",
            "evidence": []
        }

    return generate_answer_from_gemini(combined, query)
