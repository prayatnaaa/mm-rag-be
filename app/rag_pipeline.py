from app.retriever.embed_clip import embed_text_image
from app.retriever.faiss_index import search_similar_chunks
from app.llm.gemini_api import generate_answer_from_gemini
from app.db.metadata_store import get_active_sources
import hashlib


def deduplicate_contexts(contexts):
    seen = set()
    unique = []
    for ctx in contexts:
        key = hashlib.md5((ctx.get("text", "") + ctx.get("image_url", "")).encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(ctx)
    return unique


def run_rag_pipeline(query: str, image_path: str = None):
    # Ambil sumber aktif
    active_sources = get_active_sources()
    active_ids = list(active_sources.keys())
    print("Active source IDs:", active_ids)

    # Embedding teks dan gambar secara terpisah
    vecs = embed_text_image(query, image_path)
    vec_text = vecs["text_embedding"]
    vec_image = vecs["image_embedding"]

    print("Text embedding shape:", vec_text.shape)
    print("Image embedding shape:", vec_image.shape)

    # Retrieval berdasarkan masing-masing modalitas
    results_text = search_similar_chunks(vec_text, top_k=5, allowed_source_ids=active_ids)
    results_image = search_similar_chunks(vec_image, top_k=5, allowed_source_ids=active_ids)

    print(f"Retrieved {len(results_text)} text chunks, {len(results_image)} image chunks")

    # Gabungkan dan buang duplikat
    combined = deduplicate_contexts(results_text + results_image)
    combined = sorted(combined, key=lambda x: x["distance"])[:5]

    # Kirim ke LLM untuk dijawab
    return generate_answer_from_gemini(combined, query)
