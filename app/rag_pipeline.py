from app.retriever.embed_clip import embed_text_only
from app.retriever.faiss_index import search_similar_chunks
from app.llm.gemini_api import generate_answer_from_gemini

def run_rag_pipeline(query: str):
    q_vec = embed_text_only(query)
    contexts = search_similar_chunks(q_vec, top_k=5)
    return generate_answer_from_gemini(contexts, query)
