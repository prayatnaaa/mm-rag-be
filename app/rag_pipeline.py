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

    q_vec = embed_text_image(query)
    print("Query vector dim:", q_vec.shape)

    contexts = search_similar_chunks(q_vec, top_k=5, allowed_source_ids=active_ids)

    return generate_answer_from_gemini(contexts, query)
