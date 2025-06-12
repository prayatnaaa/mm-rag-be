from app.retriever.embed_clip import embed_text_only, embed_text_image
from app.retriever.faiss_index import search_similar_chunks
from app.llm.gemini_api import generate_answer_from_gemini
from app.db.metadata_store import get_active_sources
from PIL import Image

def run_rag_pipeline(query: str, image_path: str = None):
    active_sources = get_active_sources()
    active_ids = list(active_sources.keys())

    print(active_sources)
    print(image_path)

    # if image_path:
    #     print(image_path)
    #     q_vec = embed_text_image(query, image_path)
    # else:
    #     print("masuk ke text")
    #     q_vec = embed_text_only(query)
    q_vec = embed_text_image(query)
    # q_vec = embed_text_only(query)
    print("Query vector dim:", q_vec.shape)

    contexts = search_similar_chunks(q_vec, top_k=5, allowed_source_ids=active_ids)

    return generate_answer_from_gemini(contexts, query)

# def run_rag_pipeline(query: str):
#     active_sources = get_active_sources()
#     # print("active sources f", active_sources)
#     active_ids = list(active_sources.keys())
#     # print("active ids ", active_ids)

#     q_vec = embed_text_only(query)
#     contexts = search_similar_chunks(q_vec, top_k=5, allowed_source_ids=active_ids)
#     print("======================================")
#     print(contexts)
#     print("======================================")
#     # print("context from run rag pipeline ", contexts)

#     return generate_answer_from_gemini(contexts, query)

# import re

# def detect_query_mode(query: str) -> str:
#     """Deteksi apakah user ingin gambar, teks, atau campuran."""
#     query = query.lower()
#     if re.search(r"\b(gambar|foto|image|lihat|tunjukkan|menampilkan)\b", query):
#         return "image"
#     elif re.search(r"\b(teks|artikel|tulisan|penjelasan|deskripsi)\b", query):
#         return "text"
#     return "hybrid"

# def run_rag_pipeline(query: str):
#     active_sources = get_active_sources()
#     active_ids = list(active_sources.keys())

#     mode = detect_query_mode(query)
#     q_vec = embed_text_only(query)  # Text embedding tetap digunakan sebagai query anchor

#     results = search_similar_chunks(q_vec, top_k=10, allowed_source_ids=active_ids)
#     # print("===========================")
#     # print(results)
#     # print("===========================")

#     if mode == "image":
#         filtered = [r for r in results if "image_url" in r["metadata"]]
#         if not filtered:
#             return {"answer": "Tidak ditemukan gambar relevan untuk pertanyaan ini.", "evidence": []}
#     elif mode == "text":
#         print("ini saya di teks")
#         filtered = [r for r in results if "text" in r["metadata"]]
#         if not filtered:
#             return {"answer": "Tidak ditemukan informasi teks relevan.", "evidence": []}
#     else:
#         filtered = results  # Hybrid mode: campuran

#     return generate_answer_from_gemini(filtered, query)
