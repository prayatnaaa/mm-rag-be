from app.retriever.embed_clip import embed_text_only
from app.retriever.faiss_index import search_similar_chunks
from app.llm.gemini_api import generate_answer_from_gemini
from app.db.metadata_store import get_active_sources

def run_rag_pipeline(query: str):
    active_sources = get_active_sources()
    # print("active sources f", active_sources)
    active_ids = list(active_sources.keys())
    # print("active ids ", active_ids)

    q_vec = embed_text_only(query)
    contexts = search_similar_chunks(q_vec, top_k=5, allowed_source_ids=active_ids)
    # print("context from run rag pipeline ", contexts)

    return generate_answer_from_gemini(contexts, query)