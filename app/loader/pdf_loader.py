import os
from PyPDF2 import PdfReader
from app.retriever.embed_clip import embed_text_only
from app.retriever.faiss_index import add_embedding
from app.db.metadata_store import save_source
from app.utils.chunker import chunk_text  

def load_pdf_data(file_path: str, source_id: str):
    reader = PdfReader(file_path)
    all_text = []

    for page in reader.pages:
        text = page.extract_text() or ""
        all_text.append(text.strip())

    combined_text = "\n".join(all_text)
    chunks = chunk_text(combined_text, chunk_size=200, overlap=40)

    embedding_ids = []
    for chunk in chunks:
        vec = embed_text_only(chunk)
        eid = add_embedding(vec, metadata={"source": source_id, "text": chunk})
        embedding_ids.append(eid)

    save_source(source_id, file_path, f"PDF: {os.path.basename(file_path)}", embedding_ids)
    return len(embedding_ids)
