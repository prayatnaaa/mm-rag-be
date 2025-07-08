from app.retriever.chromadb_index import add_embedding, embed_text
import os
from app.db.metadata_store import save_source

def clean_metadata(meta: dict):
    return {k: v for k, v in meta.items() if v is not None}

def load_txt_data(txt_path: str, source_id: str):
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
    chunks = []
    current_text = ""
    for paragraph in paragraphs:
        if len(current_text + paragraph) < 500:
            current_text += " " + paragraph
        else:
            chunks.append(current_text.strip())
            current_text = paragraph
    if current_text:
        chunks.append(current_text.strip())

    embeddings = []
    for idx, chunk in enumerate(chunks):
        vec = embed_text(chunk)
        meta = {
            "source_id": source_id,
            "text": chunk,
            "title": os.path.basename(txt_path),
            "source": "txt",
            "active": True,
        }
        eid = add_embedding(vec, clean_metadata(meta | {"modality": "text"}))
        embeddings.append(eid)

    save_source(source_id, f"s3://{source_id}/", os.path.basename(txt_path), embeddings)
    return {"status": "ok", "chunks": len(embeddings)}
