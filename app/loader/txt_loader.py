import os
from app.retriever.embed_clip import embed_text_only
from app.retriever.faiss_index import add_embedding
from app.db.metadata_store import save_source
from app.utils.chunker import chunk_text 

def load_txt_data(file_path: str, source_id: str):
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    chunks = chunk_text(full_text, chunk_size=200, overlap=40)
    embedding_ids = []

    for chunk in chunks:
        vec = embed_text_only(chunk)
        eid = add_embedding(vec, metadata={"source": source_id, "text": chunk})
        embedding_ids.append(eid)

    save_source(source_id, file_path, f"TXT: {os.path.basename(file_path)}", embedding_ids)
    return len(embedding_ids)
