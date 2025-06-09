import os
from app.retriever.embed_clip import embed_text_only, truncate_clip_text
from app.retriever.faiss_index import add_embedding
from app.db.metadata_store import save_source
from app.utils.chunker import chunk_text 
from app.utils.minio_client import get_file_stream_from_minio

MINIO_BUCKET = os.getenv("MINIO_BUCKET", "images")

def load_txt_data(file_path: str, source_id: str):
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    chunks = truncate_clip_text(full_text)
    embedding_ids = []

    for chunk in chunks:
        # print(chunk)
        vec = embed_text_only(chunk)
        eid = add_embedding(vec, metadata={"source": source_id, "text": chunk})
        embedding_ids.append(eid)

    save_source(source_id, file_path, f"TXT: {os.path.basename(file_path)}", embedding_ids)
    return len(embedding_ids)

def load_txt_data_from_minio(object_name: str, source_id: str):
    stream = get_file_stream_from_minio(object_name)
    text = stream.read().decode("utf-8")

    chunks = truncate_clip_text(text)
    embedding_ids = []

    for chunk in chunks:
        vec = embed_text_only(chunk)
        eid = add_embedding(vec, metadata={"source": source_id, "text": chunk})
        embedding_ids.append(eid)

    save_source(source_id, f"s3://{MINIO_BUCKET}/{object_name}", f"TXT: {object_name}", embedding_ids)
    return len(embedding_ids)
